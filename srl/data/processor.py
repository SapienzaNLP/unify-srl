import json

import torch
from torch.nn.utils.rnn import pad_sequence
from allennlp.modules.elmo import batch_to_ids
from transformers import AutoTokenizer

from layers.word_encoder import RandomlyInitializedEmbedding, FastTextEmbedding
from utils.decoding import viterbi_decode


class Processor(object):

    def __init__(
            self,
            dataset={},
            input_representation='randomly_initialized_embeddings',
            vocab_size=10_000,
            lowercase=False,
            min_word_frequency=1,
            model_name='',
            padding_token='<pad>',
            unknown_token='<unk>',
            **kwargs):

        super(Processor, self).__init__()

        self.input_representation = input_representation
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.padding_target_id = -1

        if input_representation == 'randomly_initialized_embeddings':
            input_maps = RandomlyInitializedEmbedding.build_input_maps(
                dataset,
                vocab_size,
                lowercase,
                min_word_frequency,
                padding_token,
                unknown_token)
            self.word2id = input_maps['word2id']
            self.word_counter = input_maps['word_counter']
            self.padding_token_id = self.word2id[padding_token]
            self.unknown_token_id = self.word2id[unknown_token]

        elif input_representation == 'fasttext_embeddings':
            input_maps = FastTextEmbedding.build_input_maps(
                dataset,
                vocab_size,
                lowercase,
                min_word_frequency,
                padding_token,
                unknown_token)
            self.word2id = input_maps['word2id']
            self.word_counter = input_maps['word_counter']
            self.padding_token_id = self.word2id[padding_token]
            self.unknown_token_id = self.word2id[unknown_token]

        elif input_representation == 'elmo_embeddings':
            self.padding_token_id = 0
            self.unknown_token_id = 0

        elif input_representation == 'bert_embeddings':
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.padding_token_id = self.tokenizer.pad_token_id
            self.unknown_token_id = self.tokenizer.unk_token_id

        output_maps = Processor._build_output_maps(dataset)
        self.predicate2id = output_maps['predicate2id']
        self.id2predicate = output_maps['id2predicate']
        self.word2sense = output_maps['word2sense']
        self.unknown_predicate_id = self.predicate2id[unknown_token]
        self.num_senses = len(self.predicate2id)
        self.role2id = output_maps['role2id']
        self.id2role = output_maps['id2role']
        self.role_weights = output_maps['role_weights']
        self.role_transition_matrix = output_maps['role_transition_matrix']
        self.role_start_transitions = output_maps['role_start_transitions']
        self.unknown_role_id = self.role2id[unknown_token]
        self.num_roles = len(self.role2id)

    def encode_sentence(self, sentence):
        word_ids = []
        subword_indices = []
        sequence_length = len(sentence['words'])
        if self.input_representation == 'elmo_embeddings' or self.input_representation == 'bert_embeddings':
            sequence_length += 2
        tokenized_sequence_length = sequence_length

        if self.input_representation == 'randomly_initialized_embeddings' or self.input_representation == 'fasttext_embeddings':
            for word in sentence['words']:
                if self.lowercase:
                    word = word.lower()
                if word in self.word2id:
                    word_ids.append(self.word2id[word])
                else:
                    word_ids.append(self.unknown_token_id)

        elif self.input_representation == 'elmo_embeddings':
            word_ids = batch_to_ids([sentence['words']]).squeeze(0)

        elif self.input_representation == 'bert_embeddings':
            tokenized_sentence = []
            subword_indices = [1]
            for word_index, word in enumerate(sentence['words']):
                tokenized_word = self.tokenizer.tokenize(word)
                tokenized_sentence.extend(tokenized_word)
                subword_indices.extend([word_index + 2]*len(tokenized_word))
                if len(tokenized_sentence) > 500:
                    tokenized_sentence = tokenized_sentence[:500]
                    subword_indices = subword_indices[:500]
                    sequence_length = word_index + 2
                    break
            subword_indices.append(word_index + 3)
            subword_indices.append(0)
            word_ids = self.tokenizer.encode(tokenized_sentence) + [self.padding_token_id]
            tokenized_sequence_length = len(word_ids) - 1

        word_sense_mask = []
        if self.input_representation == 'bert_embeddings':
            mask = [True]*self.num_senses
            word_sense_mask.append(mask)

        for word in sentence['words']:
            lowercased_word = word.lower()
            if lowercased_word not in self.word2sense:
                mask = [True]*self.num_senses
            else:
                mask = [False]*self.num_senses
                word_sense_ids = self.word2sense[lowercased_word]
                for word_sense_id in word_sense_ids:
                    mask[word_sense_id] = True
            word_sense_mask.append(mask)
            if self.input_representation == 'bert_embeddings':
                tokenized_word = self.tokenizer.tokenize(word)
                for _ in range(len(tokenized_word) - 1):
                    word_sense_mask.append(mask)

        if self.input_representation == 'bert_embeddings':
            mask = [True]*self.num_senses
            word_sense_mask.append(mask)

        return {
            'word_ids': torch.as_tensor(word_ids),
            'subword_indices': torch.as_tensor(subword_indices),
            'sequence_length': sequence_length,
            'tokenized_sequence_length': tokenized_sequence_length,
            'word_sense_mask': torch.as_tensor(word_sense_mask),
        }

    def encode_labels(self, sentence):
        predicates = []
        senses = []
        predicate_indices = {}
        subtoken_index = 0

        if self.input_representation == 'elmo_embeddings' or self.input_representation == 'bert_embeddings':
            subtoken_index += 1
            predicates.append(self.padding_target_id)
            senses.append(self.padding_target_id)

        for word_index, (predicate, word) in enumerate(zip(sentence['predicates'], sentence['words'])):

            if predicate != '_':
                predicates.append(1)
                predicate_indices[word_index] = subtoken_index
            else:
                predicates.append(0)

            subtoken_index += 1

            if predicate in self.predicate2id:
                senses.append(self.predicate2id[predicate])
            else:
                senses.append(self.unknown_predicate_id)

        if self.input_representation == 'elmo_embeddings' or self.input_representation == 'bert_embeddings':
            predicates.append(self.padding_target_id)
            senses.append(self.padding_target_id)

        sentence_length = len(predicates)
        roles = [[self.padding_target_id] * sentence_length] * sentence_length

        for predicate_index, predicate_roles in sentence['roles'].items():
            predicate_role_ids = []
            if self.input_representation == 'elmo_embeddings' or self.input_representation == 'bert_embeddings':
                predicate_role_ids.append(self.padding_target_id)

            for role_index, role in enumerate(predicate_roles):
                if role in self.role2id:
                    predicate_role_ids.append(self.role2id[role])
                else:
                    predicate_role_ids.append(self.unknown_role_id)

            if self.input_representation == 'elmo_embeddings' or self.input_representation == 'bert_embeddings':
                predicate_role_ids.append(self.padding_target_id)

            roles[predicate_indices[predicate_index]] = predicate_role_ids

        predicate_indices = sorted(list(predicate_indices.values()))

        return {
            'predicate_indices': predicate_indices,
            'predicates': torch.as_tensor(predicates),
            'senses': torch.as_tensor(senses),
            'roles': roles,
        }

    def decode(self, x, y, viterbi_decoding=True):
        word_ids = x['word_ids']
        sentence_lengths = x['sequence_lengths']
        predicate_indices = list(map(list, zip(*x['predicate_indices'])))

        predicates = []
        predicate_ids = torch.argmax(y['predicates'], dim=-1).tolist()
        for sentence_predicate_ids, sentence_length in zip(predicate_ids, sentence_lengths):
            if self.input_representation == 'elmo_embeddings' or self.input_representation == 'bert_embeddings':
                sentence_predicate_ids = sentence_predicate_ids[1:sentence_length-1]
            else:
                sentence_predicate_ids = sentence_predicate_ids[:sentence_length]
            predicates.append([p for p in sentence_predicate_ids])

        senses = [['_']*sentence_length for sentence_length in sentence_lengths]
        sense_ids = torch.argmax(y['senses'], dim=-1).tolist()
        for (sentence_index, predicate_index), sense_id in zip(predicate_indices, sense_ids):
            if self.input_representation == 'elmo_embeddings' or self.input_representation == 'bert_embeddings':
                predicate_index -= 1
            senses[sentence_index][predicate_index] = self.id2predicate[sense_id]

        roles = {i: {} for i in range(len(word_ids))}
        if not viterbi_decoding:
            role_ids = torch.argmax(y['roles'], dim=-1).tolist()
            for (sentence_index, predicate_index), predicate_role_ids in zip(predicate_indices, role_ids):
                sentence_length = sentence_lengths[sentence_index]
                if self.input_representation == 'elmo_embeddings' or self.input_representation == 'bert_embeddings':
                    predicate_index -= 1
                    predicate_role_ids = predicate_role_ids[1:sentence_length-1]
                else:
                    predicate_role_ids = predicate_role_ids[:sentence_length]
                predicate_roles = [self.id2role[r] for r in predicate_role_ids]
                roles[sentence_index][predicate_index] = predicate_roles
        else:
            role_emissions = y['roles']
            for (sentence_index, predicate_index), predicate_role_emissions in zip(predicate_indices, role_emissions):
                sentence_length = sentence_lengths[sentence_index]
                if self.input_representation == 'elmo_embeddings' or self.input_representation == 'bert_embeddings':
                    predicate_index -= 1
                    predicate_role_emissions = predicate_role_emissions[1:sentence_length-1]
                else:
                    predicate_role_emissions = predicate_role_emissions[:sentence_length]
                predicate_role_ids, _ = viterbi_decode(
                    predicate_role_emissions.to('cpu'),
                    torch.as_tensor(self.role_transition_matrix),
                    allowed_start_transitions=torch.as_tensor(self.role_start_transitions))
                predicate_roles = [self.id2role[r] for r in predicate_role_ids]
                roles[sentence_index][predicate_index] = predicate_roles

        return {
            'predicates': predicates,
            'senses': senses,
            'roles': roles,
        }

    def collate_sentences(self, sentences):
        batched_x = {
            'sentence_ids': [],
            'predicate_indices': [[], []],

            'word_ids': [],
            'subword_indices': [],
            'sequence_lengths': [],
            'tokenized_sequence_lengths': [],
            'word_sense_mask': [],
        }

        batched_y = {
            'predicates': [],
            'senses': [],
            'roles': [],
        }

        max_sequence_length = 0
        sentence_index = 0
        for sentence_index, sentence in enumerate(sentences):
            encoded_sentence = self.encode_sentence(sentence)
            encoded_labels = self.encode_labels(sentence)

            sequence_length = encoded_sentence['sequence_length']
            max_sequence_length = max(max_sequence_length, sequence_length)

            batched_x['sentence_ids'].append(sentence['sentence_id'])
            batched_x['predicate_indices'][0].extend([sentence_index]*len(encoded_labels['predicate_indices']))
            batched_x['predicate_indices'][1].extend(encoded_labels['predicate_indices'])

            batched_x['word_ids'].append(encoded_sentence['word_ids'])
            batched_x['subword_indices'].append(encoded_sentence['subword_indices'])
            batched_x['sequence_lengths'].append(encoded_sentence['sequence_length'])
            batched_x['tokenized_sequence_lengths'].append(encoded_sentence['tokenized_sequence_length'])
            batched_x['word_sense_mask'].append(encoded_sentence['word_sense_mask'])

            batched_y['predicates'].append(encoded_labels['predicates'])
            batched_y['senses'].append(encoded_labels['senses'])
            batched_y['roles'].append(encoded_labels['roles'])

        if self.input_representation == 'randomly_initialized_embeddings' or self.input_representation == 'fasttext_embeddings':
            batched_x['word_ids'] = pad_sequence(batched_x['word_ids'], batch_first=True, padding_value=self.padding_token_id)
        elif self.input_representation == 'elmo_embeddings':
            batched_x['word_ids'] = pad_sequence(batched_x['word_ids'], batch_first=True)
        elif self.input_representation == 'bert_embeddings':
            batched_x['word_ids'] = pad_sequence(batched_x['word_ids'], batch_first=True, padding_value=self.padding_token_id)

        batched_x['sequence_lengths'] = torch.as_tensor(batched_x['sequence_lengths'])
        batched_x['tokenized_sequence_lengths'] = torch.as_tensor(batched_x['tokenized_sequence_lengths'])

        batched_x['subword_indices'] = pad_sequence(batched_x['subword_indices'], batch_first=True, padding_value=0)
        batched_x['word_sense_mask'] = pad_sequence(batched_x['word_sense_mask'], batch_first=True, padding_value=True)
        batched_y['predicates'] = pad_sequence(batched_y['predicates'], batch_first=True, padding_value=self.padding_target_id)
        batched_y['senses'] = pad_sequence(batched_y['senses'], batch_first=True, padding_value=self.padding_target_id)
        batched_y['roles'] = Processor._pad_bidimensional_sequences(batched_y['roles'], max_sequence_length, self.padding_target_id)
        batched_y['roles'] = torch.as_tensor(batched_y['roles'])

        return batched_x, batched_y

    def save_config(self, path):
        save_word_dictionaries = self.input_representation == 'randomly_initialized_embeddings' \
            or self.input_representation == 'fasttext_embeddings'

        config = {
            'input_representation': self.input_representation,
            'lowercase': self.lowercase,
            'vocab_size': self.vocab_size,
            'padding_target_id': self.padding_target_id,
            'model_name': self.model_name if self.input_representation == 'bert_embeddings' else '',

            'word2id': self.word2id if save_word_dictionaries else {},
            'word_counter': self.word_counter if save_word_dictionaries else {},
            'padding_token_id': self.padding_token_id,
            'unknown_token_id': self.unknown_token_id,

            'predicate2id': self.predicate2id,
            'id2predicate': self.id2predicate,
            'unknown_predicate_id': self.unknown_predicate_id,
            'num_senses': self.num_senses,

            'role2id': self.role2id,
            'id2role': self.id2role,
            'unknown_role_id': self.unknown_role_id,
            'num_roles': self.num_roles,

            'role_weights': self.role_weights,
            'role_transition_matrix': self.role_transition_matrix,
            'role_start_transitions': self.role_start_transitions,
            'word2sense': self.word2sense,
        }

        with open(path, 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

    @staticmethod
    def from_config(path):
        with open(path) as f:
            config = json.load(f)

        processor = Processor()
        processor.input_representation = config['input_representation']
        processor.lowercase = config['lowercase']
        processor.vocab_size = config['vocab_size']
        processor.padding_target_id = config['padding_target_id']
        processor.model_name = config['model_name']

        processor.word2id = config['word2id']
        processor.word_counter = config['word_counter']
        processor.padding_token_id = config['padding_token_id']
        processor.unknown_token_id = config['unknown_token_id']

        processor.predicate2id = config['predicate2id']
        processor.id2predicate = {int(id): predicate for id, predicate in config['id2predicate'].items()}
        processor.unknown_predicate_id = config['unknown_predicate_id']
        processor.num_senses = config['num_senses']

        processor.role2id = config['role2id']
        processor.id2role = {int(id): role for id, role in config['id2role'].items()}
        processor.unknown_role_id = config['unknown_role_id']
        processor.num_roles = config['num_roles']

        processor.role_weights = config['role_weights']
        processor.role_transition_matrix = config['role_transition_matrix']
        processor.role_start_transitions = config['role_start_transitions']
        processor.word2sense = config['word2sense']

        if processor.input_representation == 'bert_embeddings':
            processor.tokenizer = AutoTokenizer.from_pretrained(processor.model_name)

        return processor

    @staticmethod
    def _build_output_maps(dataset):
        predicate2id = {
            '_': 0,
            '<unk>': 1,
        }
        role2id = {
            '_': 0,
            '<unk>': 1
        }
        word2sense = {}

        role_counts = {}
        total_role_count = 0
        for i in range(len(dataset)):
            sentence = dataset[i]

            for word, predicate in zip(sentence['words'], sentence['predicates']):
                if predicate not in predicate2id:
                    predicate2id[predicate] = len(predicate2id)
                if predicate != '_':
                    word = word.lower()
                    if word not in word2sense:
                        word2sense[word] = set()
                    word2sense[word].add(predicate)

            for roles in sentence['roles'].values():
                for role in roles:
                    if role not in role2id:
                        role2id[role] = len(role2id)
                    role_id = role2id[role]
                    if role_id not in role_counts:
                        role_counts[role_id] = 0
                    role_counts[role_id] += 1
                    total_role_count += 1

        id2predicate = {id: predicate for predicate, id in predicate2id.items()}
        id2role = {id: role for role, id in role2id.items()}

        role_weights = [1.0]*len(role2id)
        # role_weights = [1e-3]*len(role2id)
        # # log_total_role_count = math.log(total_role_count)
        # for role_id, role_count in role_counts.items():
        #     role_weights[role_id] = ((total_role_count - role_count) / total_role_count)**0.5
        #     # role_weights[role_id] = (log_total_role_count - math.log(1 + role_counts[role_id])) / log_total_role_count

        word2sense = {word: [predicate2id[s] for s in senses] for word, senses in word2sense.items()}

        role_transition_matrix = []
        for i in range(len(id2role)):
            previous_label = id2role[i]
            role_transitions = []
            for j in range(len(id2role)):
                label = id2role[j]
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    role_transitions.append(float('-inf'))
                else:
                    role_transitions.append(0.0)
            role_transition_matrix.append(role_transitions)
        
        role_start_transitions = []
        for i in range(len(id2role)):
            label = id2role[i]
            if label[0] == "I":
                role_start_transitions.append(float("-inf"))
            else:
                role_start_transitions.append(0.0)

        return {
            'predicate2id': predicate2id,
            'id2predicate': id2predicate,
            'role2id': role2id,
            'id2role': id2role,
            'role_weights': role_weights,
            'word2sense': word2sense,
            'role_transition_matrix': role_transition_matrix,
            'role_start_transitions': role_start_transitions,
        }

    @staticmethod
    def _pad_sequences(sequences, sequence_length, padding_token_id):
        padded_sequences = []
        for sequence in sequences:
            padded_sequence = sequence + ([padding_token_id]*(sequence_length - len(sequence)))
            padded_sequences.append(padded_sequence)
        return padded_sequences

    @staticmethod
    def _pad_bidimensional_sequences(sequences, sequence_length, padding_token_id):
        padded_sequences = []
        for sequence in sequences:
            padded_sequence = []
            for subsequence in sequence:
                padded_subsequence = subsequence + ([padding_token_id]*(sequence_length - len(subsequence)))
                padded_sequence.append(padded_subsequence)
            for i in range(sequence_length - len(sequence)):
                padded_sequence.append([padding_token_id]*sequence_length)
            padded_sequences.append(padded_sequence)
        return padded_sequences

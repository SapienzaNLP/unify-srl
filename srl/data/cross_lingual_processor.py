import json

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class CrossLingualProcessor(object):

    def __init__(
            self,
            dataset={},
            model_name='bert-base-multilingual-cased',
            unknown_token='<unk>'):

        super(CrossLingualProcessor, self).__init__()

        self.padding_target_id = -1

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.padding_token_id = self.tokenizer.pad_token_id
        self.unknown_token_id = self.tokenizer.unk_token_id

        self.predicate2id, self.id2predicate, self.unknown_predicate_id, self.num_senses = dict(), dict(), dict(), dict()
        self.role2id, self.id2role, self.unknown_role_id, self.num_roles = dict(), dict(), dict(), dict()
        
        if dataset:
            self.languages = dataset.languages
            for language in self.languages:
                output_maps = CrossLingualProcessor._build_output_maps(dataset, language)
                self.predicate2id[language] = output_maps['predicate2id']
                self.id2predicate[language] = output_maps['id2predicate']
                self.unknown_predicate_id[language] = self.predicate2id[language][unknown_token]
                self.num_senses[language] = len(self.predicate2id[language])
                self.role2id[language] = output_maps['role2id']
                self.id2role[language] = output_maps['id2role']
                self.unknown_role_id[language] = self.role2id[language][unknown_token]
                self.num_roles[language] = len(self.role2id[language])

    def encode_sentence(self, sentence):
        word_ids = []
        subword_indices = []
        sequence_length = len(sentence['words']) + 2

        tokenized_sentence = []
        subword_indices = [0]
        word_index = 1
        for word_index, word in enumerate(sentence['words'], 1):
            tokenized_word = self.tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)
            subword_indices.extend([word_index]*len(tokenized_word))
            
            if len(tokenized_sentence) > 500:
                tokenized_sentence = tokenized_sentence[:500]
                subword_indices = subword_indices[:500]
                sequence_length = word_index
                break

        subword_indices.append(word_index + 1)
        word_ids = self.tokenizer.encode(tokenized_sentence)

        return {
            'word_ids': torch.as_tensor(word_ids),
            'subword_indices': torch.as_tensor(subword_indices),
            'sequence_length': sequence_length,
            'tokenized_sequence_length': len(word_ids) - 1,
        }

    def encode_labels(self, sentence):
        language = sentence['language']
        predicates = []
        senses = []
        predicate_indices = []

        for word_index, predicate in enumerate(sentence['predicates']):
            if predicate != '_':
                predicates.append(1)
                predicate_indices.append(word_index)
            else:
                predicates.append(0)

            if predicate in self.predicate2id[language]:
                senses.append(self.predicate2id[language][predicate])
            else:
                senses.append(self.unknown_predicate_id[language])

        sentence_length = len(predicates)
        roles = [[self.padding_target_id] * sentence_length] * sentence_length

        for predicate_index, predicate_roles in sentence['roles'].items():
            predicate_role_ids = []
            for role in predicate_roles:
                if role in self.role2id[language]:
                    predicate_role_ids.append(self.role2id[language][role])
                else:
                    predicate_role_ids.append(self.unknown_role_id[language])
            roles[predicate_index] = predicate_role_ids

        return {
            'predicate_indices': predicate_indices,
            'predicates': torch.as_tensor(predicates),
            'senses': torch.as_tensor(senses),
            'roles': roles,
        }

    @staticmethod
    def _decode_remove_language_prefix(label: str):
        if label == '_':
            return label
        else:
            return label[3:]

    def decode(self, x, y, language: int):
        word_ids = x['word_ids']
        sentence_lengths = x['sequence_lengths']
        predicate_indices = list(map(list, zip(*x['predicate_indices'])))

        predicates = []
        predicate_ids = torch.argmax(y['predicates'], dim=-1).tolist()
        for sentence_predicate_ids, sentence_length in zip(predicate_ids, sentence_lengths):
            sentence_predicate_ids = sentence_predicate_ids[:sentence_length]
            predicates.append([p for p in sentence_predicate_ids])

        senses = [['_'] * sentence_length for sentence_length in sentence_lengths]
        sense_ids = torch.argmax(y['senses'], dim=-1).tolist()
        for (sentence_index, predicate_index), sense_id in zip(predicate_indices, sense_ids):
            senses[sentence_index][predicate_index] = self._decode_remove_language_prefix(self.id2predicate[language][sense_id])

        roles = {i: {} for i in range(len(word_ids))}
        role_ids = torch.argmax(y['roles'], dim=-1).tolist()
        for (sentence_index, predicate_index), predicate_role_ids in zip(predicate_indices, role_ids):
            sentence_length = sentence_lengths[sentence_index]
            predicate_role_ids = predicate_role_ids[:sentence_length]
            predicate_roles = [self.id2role[language][r] for r in predicate_role_ids]
            predicate_roles = [self._decode_remove_language_prefix(r) for r in predicate_roles]
            roles[sentence_index][predicate_index] = predicate_roles

        return {
            'predicates': predicates,
            'senses': senses,
            'roles': roles,
        }

    @staticmethod
    def _get_batch_input_template(language: str) -> dict:
        return {
            'language': language,
            'sentence_ids': [],
            'predicate_indices': [[], []],

            'word_ids': [],
            'subword_indices': [],
            'sequence_lengths': [],
            'tokenized_sequence_lengths': [],
        }

    @staticmethod
    def _get_batch_output_template(language: str) -> dict:
        return {
            'language': language,
            'predicates': [],
            'senses': [],
            'roles': [],
        }

    def collate_sentences(self, sentences):
        batched_x = {}
        batched_y = {}
        max_sequence_length = {}
        sentence_index = {}

        for sentence in sentences:
            language: str = sentence['language']
            if language not in batched_x:
                batched_x[language] = CrossLingualProcessor._get_batch_input_template(language)
                batched_y[language] = CrossLingualProcessor._get_batch_output_template(language)
                max_sequence_length[language] = 0
                sentence_index[language] = 0

            encoded_sentence = self.encode_sentence(sentence)
            encoded_labels = self.encode_labels(sentence)

            max_sequence_length[language] = max(encoded_sentence['sequence_length'], max_sequence_length[language])

            batched_x[language]['sentence_ids'].append(sentence['sentence_id'])
            batched_x[language]['predicate_indices'][0].extend([sentence_index[language]]*len(encoded_labels['predicate_indices']))
            batched_x[language]['predicate_indices'][1].extend(encoded_labels['predicate_indices'])
            sentence_index[language] += 1

            batched_x[language]['word_ids'].append(encoded_sentence['word_ids'])
            batched_x[language]['subword_indices'].append(encoded_sentence['subword_indices'])
            batched_x[language]['sequence_lengths'].append(encoded_sentence['sequence_length'])
            batched_x[language]['tokenized_sequence_lengths'].append(encoded_sentence['tokenized_sequence_length'])

            batched_y[language]['predicates'].append(encoded_labels['predicates'])
            batched_y[language]['senses'].append(encoded_labels['senses'])
            batched_y[language]['roles'].append(encoded_labels['roles'])

        for language in batched_x.keys():
            batched_x[language]['word_ids'] = pad_sequence(batched_x[language]['word_ids'], batch_first=True, padding_value=self.padding_token_id)

            batched_x[language]['sequence_lengths'] = torch.as_tensor(batched_x[language]['sequence_lengths'])
            batched_x[language]['tokenized_sequence_lengths'] = torch.as_tensor(batched_x[language]['tokenized_sequence_lengths'])

            batched_x[language]['subword_indices'] = pad_sequence(
                batched_x[language]['subword_indices'],
                batch_first=True,
                padding_value=max_sequence_length[language] - 1)
            batched_y[language]['predicates'] = pad_sequence(
                batched_y[language]['predicates'],
                batch_first=True,
                padding_value=self.padding_target_id)
            batched_y[language]['senses'] = pad_sequence(
                batched_y[language]['senses'],
                batch_first=True,
                padding_value=self.padding_target_id)
            batched_y[language]['roles'] = CrossLingualProcessor._pad_bidimensional_sequences(
                batched_y[language]['roles'],
                sequence_length=max_sequence_length[language] - 2,
                padding_value=self.padding_target_id)
            
        return batched_x, batched_y

    def save_config(self, path):

        config = {
            'padding_target_id': self.padding_target_id,
            'model_name': self.model_name,

            'padding_token_id': self.padding_token_id,
            'unknown_token_id': self.unknown_token_id,

            'languages': self.languages,

            'predicate2id': self.predicate2id,
            'id2predicate': self.id2predicate,
            'unknown_predicate_id': self.unknown_predicate_id,
            'num_senses': self.num_senses,

            'role2id': self.role2id,
            'id2role': self.id2role,
            'unknown_role_id': self.unknown_role_id,
            'num_roles': self.num_roles,
        }

        with open(path, 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

    @staticmethod
    def from_config(path):
        with open(path) as f:
            config = json.load(f)

        processor = CrossLingualProcessor()
        processor.padding_target_id = config['padding_target_id']
        processor.model_name = config['model_name']

        processor.padding_token_id = config['padding_token_id']
        processor.unknown_token_id = config['unknown_token_id']

        processor.languages = config['languages']

        processor.predicate2id = config['predicate2id']
        processor.unknown_predicate_id = config['unknown_predicate_id']
        processor.num_senses = config['num_senses']

        processor.role2id = config['role2id']
        processor.unknown_role_id = config['unknown_role_id']
        processor.num_roles = config['num_roles']

        processor.id2predicate = {}
        processor.id2role = {}
        for language in processor.languages:
            _id2predicate = {int(id): predicate for predicate, id in processor.predicate2id[language].items()}
            processor.id2predicate[language] = _id2predicate

            _id2role = {int(id): role for role, id in processor.role2id[language].items()}
            processor.id2role[language] = _id2role

        processor.tokenizer = AutoTokenizer.from_pretrained(processor.model_name)

        return processor

    @staticmethod
    def _build_output_maps(dataset, language):
        predicate2id = {
            '_': 0,
            '<unk>': 1,
        }
        role2id = {
            '_': 0,
            '<unk>': 1
        }

        for i in range(len(dataset)):
            sentence = dataset[i]
            if sentence['language'] != language:
                continue

            for predicate in sentence['predicates']:
                if predicate not in predicate2id:
                    predicate2id[predicate] = len(predicate2id)

            for roles in sentence['roles'].values():
                for role in roles:
                    if role not in role2id:
                        role2id[role] = len(role2id)

        id2predicate = {id: predicate for predicate, id in predicate2id.items()}
        id2role = {id: role for role, id in role2id.items()}

        return {
            'predicate2id': predicate2id,
            'id2predicate': id2predicate,
            'role2id': role2id,
            'id2role': id2role
        }

    @staticmethod
    def _pad_bidimensional_sequences(sequences, sequence_length, padding_value=0):
        padded_sequences = torch.full((len(sequences), sequence_length, sequence_length), padding_value, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            for j, subsequence in enumerate(sequence):
                padded_sequences[i][j][:len(subsequence)] = torch.as_tensor(subsequence, dtype=torch.long)
        return padded_sequences

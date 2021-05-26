import json

from torch.utils.data import Dataset


class CoNLL(Dataset):

    def __init__(self, path_to_data):
        super().__init__()
        self.sentences = CoNLL.load_sentences(path_to_data)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return sentence

    @staticmethod
    def load_sentences(path):
        sentences = {}
        with open(path) as json_file:
            sentence_index = 0
            for i, sentence in json.load(json_file).items():
                if len(sentence['words']) > 200:
                    continue
                if not [p for p in sentence['predicates'] if p != '_']:
                    continue
                sentences[sentence_index] = {
                    'sentence_id': i,
                    'words': sentence['words'],
                    'predicates': CoNLL.process_words(sentence['predicates']),
                    'roles': {
                        int(predicate_index): roles
                        for predicate_index, roles in sentence['roles'].items()
                    },
                }
                sentence_index += 1
        return sentences

    @staticmethod
    def process_words(words):
        processed_words = []
        for word in words:
            if word == '-LRB-' or word == '-LSB-':
                processed_word = '('
            elif word == '-RRB-' or word == '-RSB-':
                processed_word = ')'
            elif word == '-LCB-' or word == '-RCB-':
                processed_word = ''
            elif word == '``' or word == "''":
                processed_word = '"'
            else:
                processed_word = word
            processed_words.append(processed_word)
        return processed_words
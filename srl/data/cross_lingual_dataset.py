import json
import random

from torch.utils.data import Dataset


class CrossLingualCoNLL(Dataset):

    def __init__(self, dataset_paths, limit=1.0):
        self.sentences: dict = {}
        start_index = 0

        for language, path in dataset_paths.items():
            language_sentences: dict = CrossLingualCoNLL.load_sentences(path, language, start_index, limit)
            start_index += len(language_sentences)
            self.sentences.update(language_sentences)

        self.languages: list = sorted(list(dataset_paths.keys()))


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return sentence

    @staticmethod
    def load_sentences(path: str, language: str, start_index: int, limit):
        sentences = {}
        num_sentences = 0

        with open(path) as json_file:
            data = json.load(json_file).items()
        
        num_annotated_sentences = 0
        for _, sentence in data:
            if len(sentence['words']) <= 200 and [p for p in sentence['predicates'] if p != '_']:
                num_annotated_sentences += 1

        if isinstance(limit, float):
            limit = int(limit * num_annotated_sentences)
        else:
            limit = min(limit, num_annotated_sentences)
        
        for sentence_id, sentence in data:
            if num_sentences >= limit:
                break
            if len(sentence['words']) > 200:
                continue
            if not [p for p in sentence['predicates'] if p != '_']:
                continue

            sentences[start_index + num_sentences] = {
                'language': language,
                'sentence_id': sentence_id,
                'words': CrossLingualCoNLL.process_words(sentence['words']),
                'predicates': CrossLingualCoNLL.prepend_language_labels(sentence['predicates'], language),
                'roles': {
                    int(predicate_index): CrossLingualCoNLL.prepend_language_labels(roles, language)
                    for predicate_index, roles in sentence['roles'].items()
                },
            }
            num_sentences += 1
        return sentences

    @staticmethod
    def process_words(words):
        processed_words = []
        for word in words:
            if word != '_':
                word = word.replace('_', ' ')
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
    
    @staticmethod
    def prepend_language_labels(labels, language, ignore_label='_'):
        new_labels = []
        for label in labels:
            if label == ignore_label:
                new_labels.append(label)
            else:
                new_label = '{}_{}'.format(language, label)
                new_labels.append(new_label)
        return new_labels

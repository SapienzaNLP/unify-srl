import json
from typing import *
import statistics
from prettytable import PrettyTable

select_dataset: dict = {
'train': {
        'en': [('../data/json/en/CoNLL2009_train.json', 'en')],
        'de': [('../data/json/de/CoNLL2009_train.json', 'de')],
        'es': [('../data/json/es/CoNLL2009_train.json', 'es')],
        'ca': [('../data/json/ca/CoNLL2009_train.json', 'ca')],
        'zh': [('../data/json/zh/CoNLL2009_train.json', 'zh')],
        'cz': [('../data/json/cz/CoNLL2009_train.json', 'cz')],
        'full_no_cz': [('../data/json/en/CoNLL2009_train.json', 'en'), ('../data/json/es/CoNLL2009_train.json', 'es'), ('../data/json/de/CoNLL2009_train.json', 'de'), ('../data/json/ca/CoNLL2009_train.json', 'ca'), ('../data/json/zh/CoNLL2009_train.json', 'zh')],
    },
    'dev': {
        'en': [('../data/json/en/CoNLL2009_dev.json', 'en')],
        'de': [('../data/json/de/CoNLL2009_dev.json', 'de')],
        'es': [('../data/json/es/CoNLL2009_dev.json', 'es')],
        'ca': [('../data/json/ca/CoNLL2009_dev.json', 'ca')],
        'zh': [('../data/json/zh/CoNLL2009_dev.json', 'zh')],
        'cz': [('../data/json/cz/CoNLL2009_dev.json', 'cz')],
        'full_no_cz': [('../data/json/en/CoNLL2009_dev.json', 'en'), ('../data/json/es/CoNLL2009_dev.json', 'es'), ('../data/json/de/CoNLL2009_dev.json', 'de'), ('../data/json/ca/CoNLL2009_dev.json', 'ca'), ('../data/json/zh/CoNLL2009_dev.json', 'zh')],
    },
    'test': {
        'en': [('../data/json/en/CoNLL2009_test.json', 'en')],
        'de': [('../data/json/de/CoNLL2009_test.json', 'de')],
        'es': [('../data/json/es/CoNLL2009_test.json', 'es')],
        'ca': [('../data/json/ca/CoNLL2009_test.json', 'ca')],
        'zh': [('../data/json/zh/CoNLL2009_test.json', 'zh')],
        'cz': [('../data/json/cz/CoNLL2009_test.json', 'cz')],
        'full_no_cz': [('../data/json/en/CoNLL2009_test.json', 'en'), ('../data/json/es/CoNLL2009_test.json', 'es'), ('../data/json/de/CoNLL2009_test.json', 'de'), ('../data/json/ca/CoNLL2009_test.json', 'ca'), ('../data/json/zh/CoNLL2009_test.json', 'zh')],
    }
}


def read_dataset(path: str):
    '''
    SOURCE CODE: Sapienza NLP group, from theirs utils.py file
    '''
    with open(path) as f:
        dataset = json.load(f)

    sentences, labels = {}, {}
    for sentence_id, sentence in dataset.items():
        sentence_id = int(sentence_id)
        sentences[sentence_id] = {
            'words': sentence['words'],
            'predicates': sentence['predicates'],
        }

        labels[sentence_id] = {
            'predicates': sentence['predicates'],
            'roles': {int(p): r for p, r in sentence['roles'].items()}
        }

    return sentences, labels


sentence, labels = read_dataset('../data/json/es/CoNLL2009_train.json')


def count_sentence(dict_dt, type):
    for lang in sorted(dict_dt[type].keys()):
        if 'full' in lang:
            continue
        sentence, labels = read_dataset(dict_dt[type][lang][0][0])
        print(dict_dt[type][lang][0][1], max(sentence.keys()))

def count_unique_predicate(dict_dt, type):
    for lang in sorted(dict_dt[type].keys()):
        if 'full' in lang:
            continue
        print()
        sentence, labels = read_dataset(dict_dt[type][lang][0][0])
        set_predicates: set = set()
        for id_sentence in labels.keys():
            set_predicates.update(set(labels[id_sentence]['predicates']))
        print(dict_dt[type][lang][0][1], len(set_predicates))


def count_unique_roles(dict_dt, type):
    for lang in sorted(dict_dt[type].keys()):
        if 'full' in lang:
            continue
        print()
        sentence, labels = read_dataset(dict_dt[type][lang][0][0])
        set_roles: set = set()
        for id_sentence in labels.keys():
            for seq_role in labels[id_sentence]['roles'].keys():
                set_roles.update(set(labels[id_sentence]['roles'][seq_role]))
        print(dict_dt[type][lang][0][1], len(set_roles))





count_sentence(select_dataset, 'train')
#count_unique_predicate(select_dataset, 'train')
#count_unique_roles(select_dataset, 'train')
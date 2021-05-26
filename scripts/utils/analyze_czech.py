import argparse
from pprint import pprint


def analyze_czech(path):
    pos_set = set()
    correct_pos, total_pos = 0, 0
    correct_gold_lemmas, correct_silver_lemmas, total_lemma_predicates = 0, 0, 0
    v_predicates = 0
    v_predicates_set = set()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            gold_pos = parts[4].strip()
            silver_pos = parts[5].strip()
            pos_set.add(gold_pos)
            total_pos += 1
            if silver_pos == gold_pos:
                correct_pos += 1

            if parts[12].strip() == 'Y':
                predicate = parts[13].strip()
                if predicate[:3] == 'v-w':
                    v_predicates += 1
                    v_predicates_set.add(predicate)
                else:
                    total_lemma_predicates += 1
                    gold_lemma = parts[2].strip()
                    silver_lemma = parts[3].strip()
                    if gold_lemma == predicate:
                        correct_gold_lemmas += 1
                    if silver_lemma == predicate:
                        correct_silver_lemmas += 1
                    # else:
                    #     print(predicate, gold_lemma, silver_lemma)

    pprint(pos_set)
    print('POS: {}/{} = {:0.2f}'.format(correct_pos, total_pos, 100.0*correct_pos/total_pos))
    print('Gold lemmas: {}/{} = {:0.2f}'.format(correct_gold_lemmas, total_lemma_predicates, 100.0*correct_gold_lemmas/total_lemma_predicates))
    print('Silver lemma: {}/{} = {:0.2f}'.format(correct_silver_lemmas, total_lemma_predicates, 100.0*correct_silver_lemmas/total_lemma_predicates))
    print('V-predicates: {}/{}'.format(v_predicates, len(v_predicates_set)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        type=str,
        default='data/txt/cz/CoNLL2009_train.txt',
        help='Path to the Czech dataset to analyze.'
    )

    args = parser.parse_args()
    analyze_czech(args.path)

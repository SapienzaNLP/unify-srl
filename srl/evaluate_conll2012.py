import os
import subprocess
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from data.dataset import CoNLL
from data.processor import Processor
from models.model import SimpleModel


def bio2conll(bio_roles):
    conll_roles = []
    for r_i, r in enumerate(bio_roles):
        if r[0] == 'B':
            if r_i == len(bio_roles) - 1 or bio_roles[r_i+1][0] == '_' or bio_roles[r_i+1][0] == 'B':
                conll_roles.append('({}*)'.format(r[2:]))
            else:
                conll_roles.append('({}*'.format(r[2:]))
        elif r[0] == 'I':
            if r_i == len(bio_roles) - 1 or bio_roles[r_i+1][0] == '_' or bio_roles[r_i+1][0] == 'B':
                conll_roles.append('*)')
            else:
                conll_roles.append('*')
        else:
            conll_roles.append('*')
    return conll_roles


def fix_spans(predicate_index, roles):
    for i in range(len(roles)):
        if i == predicate_index:
            roles[i] = 'B-V'
            continue
        
        if i != predicate_index and roles[i][2:] == 'V':
            roles[i] = '_'

        role = roles[i]
        if role == '_':
            continue

        bio = role[0]
        role = role[2:]
        if bio == 'I':
            if i == 0 or roles[i-1] == '_':
                roles[i] = 'B-{}'.format(role)
            elif roles[i-1][0] == 'I' and roles[i-1][2:] != role:
                roles[i] = roles[i-1]
    return roles


if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--scorer', type=str, default='scripts/evaluation/scorer_conll2005.pl')
    parser.add_argument('--processor', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)

    # Store the arguments in hparams.
    args = parser.parse_args()

    processor = Processor.from_config(args.processor)

    test_dataset = CoNLL(args.input)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=processor.collate_sentences)

    model = SimpleModel.load_from_checkpoint(args.model)
    model.eval()

    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0)
    trainer.test(model=model, test_dataloaders=test_dataloader)

    predictions = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    with torch.no_grad():
        for x, _ in test_dataloader:
            x = {k: v.to(device) if not isinstance(v, list) else v for k, v in x.items()}
            y = model(x)
            batch_predictions = processor.decode(x, y)

            for i, sentence_id in enumerate(x['sentence_ids']):
                predictions[sentence_id] = {
                    'predicates': batch_predictions['predicates'][i],
                    'senses': batch_predictions['senses'][i],
                    'roles': batch_predictions['roles'][i],
                }

    gold_path = os.path.join(args.output_dir, 'gold.txt')
    pred_path = os.path.join(args.output_dir, 'pred.txt')
    compare_path = os.path.join(args.output_dir, 'compare.txt')

    with open(gold_path, 'w') as gold_out, open(pred_path, 'w') as pred_out, open(compare_path, 'w') as compare_out:
        for i in range(len(test_dataset)):
            x = test_dataset[i]
            sentence_id = x['sentence_id']
            sentence_words = x['words']
            sentence_predicates = x['predicates']
            if sentence_id not in predictions:
                for predicate in sentence_predicates:
                    gold_out.write('{}\n'.format(predicate))
                    pred_out.write('-\n')
                gold_out.write('\n')
                pred_out.write('\n')
                continue

            predicted_roles = predictions[sentence_id]['roles']
            gold_roles = []
            output_roles = []
            for predicate_index, predicate in enumerate(sentence_predicates):
                if predicate != '_':
                    predicate_roles = fix_spans(predicate_index, predicted_roles[predicate_index])
                    predicate_roles = bio2conll(predicate_roles)
                    output_roles.append(predicate_roles)

                    predicate_roles = bio2conll(x['roles'][predicate_index])
                    gold_roles.append(predicate_roles)

            _predicate_index = 0
            predicted_predicates = predictions[sentence_id]['senses']
            for predicate_index, (predicate, predicted_predicate) in enumerate(zip(sentence_predicates, predicted_predicates)):
                if predicate != '_':
                    compare_out.write('{}\t{}\t{}\t{}\n'.format(sentence_id, predicate_index, predicate, predicted_predicate))
                    for word, r_gold, r_pred in zip(sentence_words, gold_roles[_predicate_index], output_roles[_predicate_index]):
                        compare_out.write('{}\t{}\t{}\n'.format(word, r_gold, r_pred))
                    compare_out.write('\n')
                    _predicate_index += 1

            output_roles = list(map(list, zip(*output_roles)))
            gold_roles = list(map(list, zip(*gold_roles)))
            for predicate, predicted_predicate_roles, gold_predicate_roles in zip(sentence_predicates, output_roles, gold_roles):
                predicate = predicate.strip()
                if predicate == '_':
                    predicate = '-'
                pred_line = '{}\t{}\n'.format(predicate, '\t'.join(predicted_predicate_roles))
                gold_line = '{}\t{}\n'.format(predicate, '\t'.join(gold_predicate_roles))
                pred_out.write(pred_line)
                gold_out.write(gold_line)
            pred_out.write('\n')
            gold_out.write('\n')

    subprocess.run(['perl', args.scorer, gold_path, pred_path])

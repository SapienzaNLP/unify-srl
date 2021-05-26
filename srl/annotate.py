import json
import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from tqdm import tqdm

from data.cross_lingual_dataset import CrossLingualCoNLL
from data.cross_lingual_processor import CrossLingualProcessor
from models.cross_lingual_model import CrossLingualModel


def bio2conll(bio_roles):
    conll_roles = []
    for r_i, r in enumerate(bio_roles):
        if r[0] == 'B':
            if r_i == len(bio_roles) - 1 or bio_roles[r_i + 1][0] == '_' or bio_roles[r_i + 1][0] == 'B':
                conll_roles.append('({}*)'.format(r[2:]))
            else:
                conll_roles.append('({}*'.format(r[2:]))
        elif r[0] == 'I':
            if r_i == len(bio_roles) - 1 or bio_roles[r_i + 1][0] == '_' or bio_roles[r_i + 1][0] == 'B':
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
            if i == 0 or roles[i - 1] == '_':
                roles[i] = 'B-{}'.format(role)
            elif roles[i - 1][0] == 'I' and roles[i - 1][2:] != role:
                roles[i] = roles[i - 1]
    return roles


if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--processor', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Add data args.
    parser.add_argument("--config_path", type=str, required=True)

    # Store the arguments in hparams.
    args = parser.parse_args()

    processor = CrossLingualProcessor.from_config(args.processor)
    # test_dataset = CrossLingualCoNLL([(args.input, args.input.split("/")[7])], percentage=100)
    # test_dataset = CrossLingualCoNLL([(args.input, args.input.split("/")[7])])

    with open(args.config_path) as config_f:
        config = json.load(config_f)

    test_dataset = CrossLingualCoNLL(config["test"])

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=processor.collate_sentences)

    model = CrossLingualModel.load_from_checkpoint(args.model)
    model.eval()

    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0)

    predictions = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    with torch.no_grad():
        for x, _ in test_dataloader:
            for language in x:
                _x = {k: v.to(device) if not (isinstance(v, list) or isinstance(v, int) or isinstance(v, str)) else v for k, v in x[language].items()}
                _y = model(_x, language)
                batch_predictions = processor.decode(_x, _y, language)

                if language not in predictions:
                    predictions[language] = {}

                for i, sentence_id in enumerate(_x['sentence_ids']):
                    predictions[language][sentence_id] = {
                        'predicates': batch_predictions['predicates'][i],
                        'senses': batch_predictions['senses'][i],
                        'roles': batch_predictions['roles'][i],
                    }

    with open(args.output, 'w') as f:
        json.dump(predictions, f, sort_keys=True, indent=4)
    print("Done")
    import sys

    sys.exit(0)

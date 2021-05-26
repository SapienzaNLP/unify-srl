import json
import os
import subprocess
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from data.cross_lingual_dataset import CrossLingualCoNLL
from data.cross_lingual_processor import CrossLingualProcessor
from models.cross_lingual_model import CrossLingualModel


if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--scorer', type=str, default='scripts/evaluation/scorer_conll2009.pl')
    parser.add_argument('--processor', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)

    # Store the arguments in hparams.
    args = parser.parse_args()

    with open(args.config) as config_f:
        config = json.load(config_f)

    processor = CrossLingualProcessor.from_config(args.processor)

    test_dataset = CrossLingualCoNLL(config['test'])

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=processor.collate_sentences)

    model = CrossLingualModel.load_from_checkpoint(args.model)
    model.eval()

    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0)
    trainer.test(model=model, test_dataloaders=test_dataloader)

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

    for language in config['test']:
        input_path = config['conll'][language]
        output_path = os.path.join(args.output_dir, '{}_predictions.txt'.format(language))

        print('\n'*5)
        print('Evaluation on {}'.format(language))

        sentence_id = 0
        sentence_output = []
        sentence_senses = []
        with open(input_path) as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    str_sentence_id = str(sentence_id)
                    if str_sentence_id not in predictions[language]:
                        for i in range(len(sentence_output)):
                            output_line = '\t'.join(sentence_output[i])
                            f_out.write('{}\t_\n'.format(output_line))
                        f_out.write('\n')
                        sentence_id += 1
                        sentence_output = []
                        sentence_senses = []
                        continue

                    predicted_senses = predictions[language][str_sentence_id]['senses']
                    output_senses = []
                    for predicate_index, (gold, predicted) in enumerate(zip(sentence_senses, predicted_senses)):
                        if gold != '_':
                            if language == 'CZ' and predicted == 'lemma':
                                output_senses.append(sentence_output[predicate_index][3])
                            else:
                                output_senses.append(predicted)
                        else:
                            output_senses.append('_')

                    predicted_roles = predictions[language][str_sentence_id]['roles']
                    output_roles = []
                    for i in range(len(sentence_senses)):
                        if output_senses[i] != '_':
                            output_roles.append(predicted_roles[i])

                    output_roles = list(map(list, zip(*output_roles)))
                    for i in range(len(sentence_output)):
                        if output_roles:
                            line_parts = sentence_output[i] + [output_senses[i]] + output_roles[i]
                        else:
                            line_parts = sentence_output[i] + [output_senses[i]]
                        output_line = '\t'.join(line_parts)
                        f_out.write('{}\n'.format(output_line))
                    f_out.write('\n')

                    sentence_id += 1
                    sentence_output = []
                    sentence_senses = []
                    continue

                parts = line.split('\t')
                sentence_output.append(parts[:13])
                sentence_senses.append(parts[13])


        result = subprocess.run(['perl', args.scorer, '-g', input_path, '-s', output_path, '-q'], capture_output=True)
        output = []
        for line in result.stdout.splitlines():
            line = str(line)
            if 'SEMANTIC' in line or 'Labeled precision' in line or 'Labeled recall' in line or 'Labeled F1' in line:
                output.append(line)
        
        print('\n'.join(output))


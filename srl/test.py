from argparse import ArgumentParser
# --processor /home/andrea/cross-lingual-srl/checkpoints/cross-bert-multilingualNOCecoNO_finetuning/processor_config.json --model /home/andrea/cross-lingual-srl/checkpoints/cross-bert-multilingualNOCecoNO_finetuning/checkpoint_epoch=027-val_f1=0.7791.ckpt
# --processor /home/andrea/cross-lingual-srl/checkpoints/cross-bert--DE_ONLY--no_finetuning/processor_config.json --model /home/andrea/cross-lingual-srl/checkpoints/cross-bert--DE_ONLY--no_finetuning/checkpoint_epoch=015-val_f1=0.8137.ckpt
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from data.cross_lingual_dataset import CrossLingualCoNLL
from data.cross_lingual_processor import CrossLingualProcessor
# from data.dataset import CoNLL
# from data.processor import Processor
from models.cross_lingual_model import CrossLingualModel
from models.model import SimpleModel
from utils.utilities import check_argument_dataset

select_dataset = {
    'dev': {
        'en': [('data/json/en/CoNLL2009_dev.json', 'en')],
        'de': [('data/json/de/CoNLL2009_dev.json', 'de')],
        'es': [('data/json/es/CoNLL2009_dev.json', 'es')],
        'ca': [('data/json/ca/CoNLL2009_dev.json', 'ca')],
        'zh': [('data/json/zh/CoNLL2009_dev.json', 'zh')],
        'cz': [('data/json/cz/CoNLL2009_dev.json', 'cz')],
        'full_no_cz': [('data/json/en/CoNLL2009_dev.json', 'en'), ('data/json/es/CoNLL2009_dev.json', 'es'), ('data/json/de/CoNLL2009_dev.json', 'de'), ('data/json/ca/CoNLL2009_dev.json', 'ca'), ('data/json/zh/CoNLL2009_dev.json', 'zh')],
    },
    'test': {
        'en': [('data/json/en/CoNLL2009_test.json', 'en')],
        'de': [('data/json/de/CoNLL2009_test.json', 'de')],
        'es': [('data/json/es/CoNLL2009_test.json', 'es')],
        'ca': [('data/json/ca/CoNLL2009_test.json', 'ca')],
        'zh': [('data/json/zh/CoNLL2009_test.json', 'zh')],
        'cz': [('data/json/cz/CoNLL2009_test.json', 'cz')],
        'full_no_cz': [('data/json/en/CoNLL2009_test.json', 'en'), ('data/json/es/CoNLL2009_test.json', 'es'), ('data/json/de/CoNLL2009_test.json', 'de'), ('data/json/ca/CoNLL2009_test.json', 'ca'), ('data/json/zh/CoNLL2009_test.json', 'zh')],
    }
}

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--test', type=str, default='data/json/en/CoNLL2009_test.json')
    parser.add_argument('--processor', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)


    # Store the arguments in hparams.
    args = parser.parse_args()


    processor = CrossLingualProcessor.from_config(args.processor)


    #test_dataset_used = ([(args.test, args.test.split('/')[2])])
    test_dataset_used = select_dataset['test']['full_no_cz']

    test_dataset = CrossLingualCoNLL(test_dataset_used)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=processor.collate_sentences)

    model = CrossLingualModel.load_from_checkpoint(args.model)
    model.eval()

    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0)
    trainer.test(model=model, test_dataloaders=test_dataloader)

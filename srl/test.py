#!/usr/bin/env python3

# python3 srl/test.py --model  xlmr_ft_full_all/checkpoint_epoch=026-val_f1=0.9028.ckpt  --processor xlmr_ft_full_all/processor_config.json --config_path config/full/en_config.json

import json
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from data.cross_lingual_dataset import CrossLingualCoNLL
from data.cross_lingual_processor import CrossLingualProcessor
from models.cross_lingual_model import CrossLingualModel


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument("--config_path", type=str, default="config/full/en_config.json")
    parser.add_argument("--processor", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    # Add dataloader args.
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    # Store the arguments in hparams.
    args = parser.parse_args()

    processor = CrossLingualProcessor.from_config(args.processor)

    # read the dataset config
    with open(args.config_path) as config_f:
        config = json.load(config_f)

    test_dataset = CrossLingualCoNLL(config["test"])

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=processor.collate_sentences
    )

    model = CrossLingualModel.load_from_checkpoint(args.model)
    model.eval()

    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0)
    trainer.test(model=model, test_dataloaders=test_dataloader)

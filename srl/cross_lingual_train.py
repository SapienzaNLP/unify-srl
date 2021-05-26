import json
import os

from argparse import ArgumentParser
from torch.cuda import get_device_name
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from data.cross_lingual_processor import CrossLingualProcessor
from data.cross_lingual_dataset import CrossLingualCoNLL
from models.cross_lingual_model import CrossLingualModel

if __name__ == "__main__":
    parser = ArgumentParser()

    # Add trial name.
    parser.add_argument("--name", type=str, required=True)

    # Add seed arg.
    parser.add_argument("--seed", type=int, default=313)

    # Add data args.
    parser.add_argument("--config_path", type=str, required=True)

    # Add dataloader args.
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=8)

    # Add checkpoint args.
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Add model-specific args.
    parser = CrossLingualModel.add_model_specific_args(parser)

    # Add all the available trainer options to argparse.
    parser = Trainer.add_argparse_args(parser)

    gpu_name: str = get_device_name(0)
    # precision: int = 32 if 'TITAN' in gpu_name else 16
    precision: int = 16

    parser.set_defaults(
        min_epochs=1,
        max_epochs=30,
        gpus=1,
        precision=precision,
        gradient_clip_val=1.0,
        row_log_interval=128,
        deterministic=True,
    )

    # Store the arguments in hparams.
    hparams = parser.parse_args()

    seed_everything(hparams.seed)

    with open(hparams.config_path) as config_f:
        config = json.load(config_f)

    train_dataset = CrossLingualCoNLL(config["train"], limit=config["limit_train_set"])
    dev_dataset = CrossLingualCoNLL(config["dev"])

    print(f"  Training on these languages:\n\t\t{train_dataset.languages}")
    processor = CrossLingualProcessor(train_dataset, model_name=hparams.language_model)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=hparams.shuffle,
        num_workers=hparams.num_workers,
        collate_fn=processor.collate_sentences,
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        collate_fn=processor.collate_sentences,
    )

    # Additional hparams.
    hparams.steps_per_epoch = (
            int(len(train_dataset) / (hparams.batch_size * hparams.accumulate_grad_batches))
            + 1
    )
    hparams.num_roles = processor.num_roles
    hparams.num_senses = processor.num_senses
    hparams.languages = processor.languages

    model = CrossLingualModel(hparams, padding_token_id=processor.padding_token_id)

    model_dir = os.path.join(hparams.checkpoint_dir, hparams.name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    processor_config_path = os.path.join(model_dir, "processor_config.json")
    model_checkpoint_path = os.path.join(
        model_dir, "checkpoint_{epoch:03d}-{val_f1:0.4f}"
    )

    processor.save_config(processor_config_path)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path, monitor="val_f1", mode="max"
    )

    trainer = Trainer.from_argparse_args(
        hparams, checkpoint_callback=checkpoint_callback
    )


    trainer.fit(
        model, train_dataloader=train_dataloader, val_dataloaders=dev_dataloader
    )

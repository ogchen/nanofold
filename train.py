import argparse
import configparser
import logging
import os
import torch
from pathlib import Path

from nanofold.training.chain_dataset import ChainDataset
from nanofold.training.logging import Logger
from nanofold.training.logging import MLFlowLogger
from nanofold.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file for training")
    parser.add_argument(
        "-i", "--input", help="Input chain training data in Arrow IPC file format", type=Path
    )
    parser.add_argument("-l", "--logging", help="Logging level", default="INFO")
    parser.add_argument("--mlflow", help="Log to MLFlow", action="store_true")

    return parser.parse_args()


def load_config(filepath):
    config = configparser.ConfigParser()
    with open(filepath) as f:
        config.read_file(f)
    if config.get("General", "device") == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return config


def get_dataloaders(args, config):
    train_data, test_data = ChainDataset.construct_datasets(
        args.input,
        config.getfloat("General", "train_split"),
        config.getint("General", "residue_crop_size"),
        config.getint("General", "num_msa"),
    )
    eval_dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_data, batch_size=config.getint("General", "eval_batch_size")
        ),
        "test": torch.utils.data.DataLoader(
            test_data, batch_size=config.getint("General", "eval_batch_size")
        ),
    }
    return (
        torch.utils.data.DataLoader(train_data, batch_size=config.getint("General", "batch_size")),
        eval_dataloaders,
    )


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.logging.upper()))
    config = load_config(args.config)
    torch.set_default_device(config.get("General", "device"))
    train_loader, eval_loaders = get_dataloaders(args, config)
    loggers = [
        Logger(),
    ]
    if args.mlflow:
        loggers.append(
            MLFlowLogger(
                uri=os.getenv("MLFLOW_SERVER_URI"),
                pip_requirements="requirements/requirements.train.txt",
            )
        )
    trainer = Trainer(config, loggers, log_every_n_epoch=100)
    trainer.fit(train_loader, eval_loaders, config.getint("General", "max_epoch"))


if __name__ == "__main__":
    main()

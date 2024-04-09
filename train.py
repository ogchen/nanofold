import argparse
import json
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
    with open(filepath) as f:
        params = json.load(f)
    if params["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return params


def get_dataloaders(args, params):
    train_data, test_data = ChainDataset.construct_datasets(
        args.input,
        params["train_split"],
        params["residue_crop_size"],
        params["num_msa"],
    )
    eval_dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_data,
            batch_size=params["eval_batch_size"],
            pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_data,
            batch_size=params["eval_batch_size"],
            pin_memory=True,
        ),
    }
    return (
        torch.utils.data.DataLoader(
            train_data,
            batch_size=params["batch_size"],
            pin_memory=True,
            num_workers=4,
        ),
        eval_dataloaders,
    )


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.logging.upper()))
    params = load_config(args.config)
    train_loader, eval_loaders = get_dataloaders(args, params)
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
    trainer = Trainer(params, loggers, log_every_n_epoch=100)
    trainer.fit(train_loader, eval_loaders, params["max_epoch"])


if __name__ == "__main__":
    main()

import argparse
import json
import logging
import os
import torch
from pathlib import Path

from nanofold.train.chain_dataset import ChainDataset
from nanofold.train.checkpoint_loader import CheckpointLoader
from nanofold.train.logging import Logger
from nanofold.train.logging import MLFlowLogger
from nanofold.train.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument("-c", "--config", help="Configuration file for training")
    config.add_argument("-r", "--runid", help="Resume training of run identified by MLFlow ID")
    parser.add_argument(
        "-e",
        "--epoch",
        help="Optional epoch for which to resume training. Use with --runid",
        type=int,
        required=False,
    )
    parser.add_argument("--max-epoch", help="Max number of epochs to train for", type=int)
    parser.add_argument(
        "--mlflow-log-freq", help="Log to MLFlow every n epochs", type=int, default=5
    )
    parser.add_argument("--log-freq", help="Log to stdout every n epochs", type=int, default=5)
    parser.add_argument(
        "--checkpoint-freq", help="Checkpoint every n epochs", type=int, default=100
    )
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
    return (
        torch.utils.data.DataLoader(
            train_data,
            batch_size=None,
            pin_memory=True,
            num_workers=4,
        ),
        torch.utils.data.DataLoader(
            test_data,
            batch_size=None,
            pin_memory=True,
            num_workers=1,
        ),
    )


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=getattr(logging, args.logging.upper()),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    mlflow_uri = os.getenv("MLFLOW_SERVER_URI")

    if args.runid:
        checkpoint_loader = CheckpointLoader(mlflow_uri, run_id=args.runid)
        params = checkpoint_loader.get_params()
        checkpoint = checkpoint_loader.get_checkpoint(epoch=args.epoch)
        run_id = checkpoint_loader.get_run_id() if not args.epoch else None
    else:
        params = load_config(args.config)
        checkpoint = None
        run_id = None

    loggers = [Logger(log_every_n_epoch=args.log_freq)]
    if args.mlflow:
        loggers.append(
            MLFlowLogger(
                uri=mlflow_uri,
                pip_requirements="requirements/requirements.train.txt",
                log_every_n_epoch=args.mlflow_log_freq,
                run_id=run_id,
            )
        )

    train_loader, test_loader = get_dataloaders(args, params)
    trainer = Trainer(
        params,
        loggers,
        checkpoint_save_freq=args.checkpoint_freq,
        checkpoint=checkpoint,
    )
    trainer.fit(train_loader, test_loader, args.max_epoch)


if __name__ == "__main__":
    main()
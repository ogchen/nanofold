import argparse
import configparser
import logging
import mlflow
import os
import tempfile
import torch
import torchinfo
from pathlib import Path

from nanofold.training.chain_dataset import ChainDataset
from nanofold.training.frame import Frame
from nanofold.training.model.input import InputEmbedding
from nanofold.training.model.structure import StructureModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file for training")
    parser.add_argument("-f", "--fasta", help="File containing FASTA sequences")
    parser.add_argument(
        "-i", "--input", help="Input chain training data in Arrow IPC file format", type=Path
    )
    parser.add_argument("-l", "--logging", help="Logging level", default="INFO")

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
    )
    return torch.utils.data.DataLoader(
        train_data, batch_size=config.getint("General", "batch_size")
    ), torch.utils.data.DataLoader(test_data, batch_size=config.getint("General", "batch_size"))


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.logging.upper()))
    config = load_config(args.config)
    train_loader, test_loader = get_dataloaders(args, config)

    mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_SERVER_URI"))

    input_embedder = InputEmbedding.from_config(config)
    params = StructureModule.get_args(config)
    model = StructureModule(**params)
    model = model.to(config.get("General", "device"))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.getfloat("Optimizer", "learning_rate"),
        betas=(config.getfloat("Optimizer", "beta1"), config.getfloat("Optimizer", "beta2")),
        eps=config.getfloat("Optimizer", "eps"),
    )

    with mlflow.start_run():
        mlflow.log_params(params)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_summary = Path(tmp_dir) / "model_summary.txt"
            model_summary.write_text(str(torchinfo.summary(model, verbose=0)))
            mlflow.log_artifact(model_summary)

        epoch = 0
        for batch in train_loader:
            if epoch == 100:
                break
            pair_representations = input_embedder(batch["target_feat"], batch["positions"])
            single_representations = torch.zeros(
                *batch["positions"].shape, config.getint("General", "single_embedding_size")
            )
            coords, fape_loss, aux_loss = model(
                single_representations,
                pair_representations,
                batch["local_coords"],
                Frame(
                    rotations=batch["rotations"],
                    translations=batch["translations"],
                ),
            )
            loss = fape_loss + aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mlflow.log_metric("fape_loss", fape_loss.detach().item(), step=epoch)
            mlflow.log_metric("aux_loss", aux_loss.detach().item(), step=epoch)
            mlflow.log_metric("total_loss", loss.detach().item(), step=epoch)
            epoch += 1

        mlflow.pytorch.log_model(
            model, "model", pip_requirements="requirements/requirements.train.txt"
        )


if __name__ == "__main__":
    main()

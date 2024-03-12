import argparse
import configparser
import mlflow
import os
import tempfile
import torch
import torchinfo
from pathlib import Path

from nanofold.training.mmcif import list_available_mmcif
from nanofold.training.mmcif import parse_chains
from nanofold.training.mmcif import load_model
from nanofold.training.mmcif import EmptyChainError
from nanofold.training.util import accept_chain
from nanofold.training.util import crop_chain
from nanofold.training.util import randint
from nanofold.training.model.input import encode_one_hot
from nanofold.training.model.input import InputEmbedding
from nanofold.training.model.structure import StructureModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file for training")
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files")
    parser.add_argument("-f", "--fasta", help="File containing FASTA sequences")
    return parser.parse_args()


def load_config(filepath):
    config = configparser.ConfigParser()
    with open(filepath) as f:
        config.read_file(f)
    return config


def load(filepath):
    model = load_model(filepath)
    return parse_chains(model)


def get_next_chain(files, max_iter, crop_size=32):
    epoch = 0
    while epoch < max_iter:
        index = randint(0, len(files))
        try:
            chains = load(files[index])
        except EmptyChainError as e:
            continue
        chain = chains[randint(0, len(chains))]
        if accept_chain(chain):
            chain = crop_chain(chain, crop_size)
            yield chain
            epoch += 1


def main():
    args = parse_args()
    config = load_config(args.config)
    if config.get("General", "device") == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_SERVER_URI"))
    available = list_available_mmcif(args.mmcif)
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
            model_summary.write_text(str(torchinfo.summary(model)))
            mlflow.log_artifact(model_summary)

        epoch = 0
        for chain in get_next_chain(available, max_iter=10):
            target_feat = encode_one_hot(chain.sequence)
            pair_representations = input_embedder(target_feat, torch.tensor(chain.positions))
            single_representations = torch.zeros(
                len(chain.sequence), config.getint("Other", "single_embedding_size")
            )
            coords, fape_loss, aux_loss = model(
                single_representations, pair_representations, chain.sequence, chain.frames
            )
            loss = fape_loss + aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mlflow.log_metric("fape_loss", fape_loss.detach().item(), step=epoch)
            mlflow.log_metric("aux_loss", aux_loss.detach().item(), step=epoch)
            mlflow.log_metric("total_loss", loss.detach().item(), step=epoch)
            epoch += 1

        mlflow.pytorch.log_model(model, "model", pip_requirements="requirements.train.txt")


if __name__ == "__main__":
    main()

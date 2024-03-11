import argparse
import configparser
import torch
from nanofold.mmcif import list_available_mmcif
from nanofold.mmcif import parse_chains
from nanofold.mmcif import load_model
from nanofold.mmcif import EmptyChainError
from nanofold.util import accept_chain
from nanofold.util import crop_chain
from nanofold.util import randint
from nanofold.model.input import encode_one_hot
from nanofold.model.input import InputEmbedding
from nanofold.model.structure import StructureModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file for training")
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files")
    parser.add_argument("-f", "--fasta", help="File containing FASTA sequences")
    return parser.parse_args()


def load_config(filepath):
    config = configparser.ConfigParser()
    config.read(filepath)
    return config


def load(filepath):
    model = load_model(filepath)
    return parse_chains(model)


def get_next_chain(files, crop_size=32):
    while True:
        index = randint(0, len(files))
        try:
            chains = load(files[index])
        except EmptyChainError as e:
            print(e)
            continue
        chain = chains[randint(0, len(chains))]
        if accept_chain(chain):
            chain = crop_chain(chain, crop_size)
            yield chain


def main():
    args = parse_args()
    config = load_config(args.config)
    available = list_available_mmcif(args.mmcif)
    input_embedder = InputEmbedding.from_config(config)
    model = StructureModule.from_config(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.getfloat("Optimizer", "learning_rate"),
        betas=(config.getfloat("Optimizer", "beta1"), config.getfloat("Optimizer", "beta2")),
        eps=config.getfloat("Optimizer", "eps"),
    )
    for chain in get_next_chain(available):
        target_feat = encode_one_hot(chain.sequence)
        pair_representations = input_embedder(target_feat, torch.tensor(chain.positions))
        single_representations = torch.zeros(
            len(chain.sequence), config.getint("Other", "single_embedding_size")
        )
        coords, fape_loss, aux_loss = model(
            single_representations, pair_representations, chain.sequence, chain.frames
        )
        loss = fape_loss + aux_loss
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()

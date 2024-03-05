import argparse
import torch
from nanofold.mmcif import list_available_mmcif
from nanofold.mmcif import parse_chains
from nanofold.mmcif import load_model
from nanofold.mmcif import EmptyChainError
from nanofold.util import accept_chain
from nanofold.util import crop_chain
from nanofold.util import randint
from nanofold.input import encode_one_hot
from nanofold.input import InputEmbedder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files")
    parser.add_argument("-f", "--fasta", help="File containing FASTA sequences")
    return parser.parse_args()


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
    available = list_available_mmcif(args.mmcif)
    input_embedder = InputEmbedder(embedding_size=8, position_bins=4)
    for chain in get_next_chain(available):
        target_feat = encode_one_hot(chain.sequence)
        pair_representations = input_embedder(target_feat, torch.tensor(chain.positions))
        print(pair_representations.shape)


if __name__ == "__main__":
    main()

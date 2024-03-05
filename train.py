import argparse
import torch
from nanofold.mmcif import list_available_mmcif
from nanofold.mmcif import parse_chains
from nanofold.mmcif import load_model
from nanofold.mmcif import EmptyChainError
from nanofold.util import accept_chain
from nanofold.util import crop_chain
from nanofold.util import randint


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
            print(f"Accepted chain {chain.id}")
            chain = crop_chain(chain, crop_size)
            print(len(chain))
            yield chain
        else:
            print(f"Rejected chain {chain.id}")


def main():
    args = parse_args()
    available = list_available_mmcif(args.mmcif)
    for chain in get_next_chain(available):
        pass


if __name__ == "__main__":
    main()

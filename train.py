import argparse
import torch

from nanofold.mmcif import list_available_mmcif
from nanofold.mmcif import parse_chains
from nanofold.mmcif import load_model
from nanofold.mmcif import EmptyChainError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files")
    parser.add_argument("-f", "--fasta", help="File containing FASTA sequences")
    return parser.parse_args()


def load(filepath):
    model = load_model(filepath)
    return parse_chains(model)


def get_next_chain(files):
    while True:
        index = torch.randint(0, len(files), (1,)).item()
        try:
            chains = load(files[index])
        except EmptyChainError as e:
            print(e)
            continue
        chain_index = torch.randint(0, len(chains), (1,)).item()
        if accept_chain(chains[chain_index]):
            print(f"Accepted chain {chains[chain_index].id}")
            yield chains[chain_index]
        else:
            print(f"Rejected chain {chains[chain_index].id}")


def accept_chain(chain):
    prob = 1 / 512 * max(min(len(chain), 512), 256)
    if torch.rand(1) < prob:
        return True


def main():
    args = parse_args()
    available = list_available_mmcif(args.mmcif)
    for chain in get_next_chain(available):
        pass


if __name__ == "__main__":
    main()

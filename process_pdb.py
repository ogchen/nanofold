import argparse

from nanofold.data_processing.mmcif import list_available_mmcif
from nanofold.data_processing.mmcif import load_model
from nanofold.data_processing.mmcif import parse_chains


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files")
    return parser.parse_args()


def load(filepath):
    model = load_model(filepath)
    return parse_chains(model)


def main():
    args = parse_args()
    available = list_available_mmcif(args.mmcif)
    chains = load(available[0])

    return


if __name__ == "__main__":
    main()

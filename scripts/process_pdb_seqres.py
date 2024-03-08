import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="PDB FASTA file to process")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.filepath, "r") as f:
        lines = f.readlines()

    with open(args.filepath, "w") as f:
        for id, seq in zip(lines[::2], lines[1::2]):
            if "mol:protein" in id:
                f.write(id)
                f.write(seq)


if __name__ == "__main__":
    main()

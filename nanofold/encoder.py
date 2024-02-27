import torch

RESIDUE_MAP = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
}


def encode_one_hot(seq):
    """Convert a sequence to one-hot encoding."""
    seq = seq.upper()
    one_hot = torch.zeros(len(seq), len(RESIDUE_MAP))
    for i, residue in enumerate(seq):
        one_hot[i, RESIDUE_MAP[residue]] = 1
    return one_hot

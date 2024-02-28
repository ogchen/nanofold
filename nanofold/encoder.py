import torch
from nanofold.residue import RESIDUE_LIST


def encode_one_hot(seq):
    """Convert a sequence to one-hot encoding."""
    res_map = {res[0]: i for i, res in enumerate(RESIDUE_LIST)}
    seq = seq.upper()
    one_hot = torch.zeros(len(seq), len(res_map))
    for i, residue in enumerate(seq):
        one_hot[i, res_map[residue]] = 1
    return one_hot

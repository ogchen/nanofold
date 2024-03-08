import torch
from torch import nn
from nanofold.residue import RESIDUE_LIST


def encode_one_hot(seq):
    """Convert a sequence to one-hot encoding."""
    res_map = {res[0]: i for i, res in enumerate(RESIDUE_LIST)}
    seq = seq.upper()
    one_hot = torch.zeros(len(seq), len(res_map))
    for i, residue in enumerate(seq):
        one_hot[i, res_map[residue]] = 1
    return one_hot


class InputEmbedding(nn.Module):
    def __init__(self, embedding_size, position_bins):
        super().__init__()
        self.embedding_size = embedding_size
        self.bins = position_bins
        input_size = len(RESIDUE_LIST)
        self.linear_a = nn.Linear(input_size, self.embedding_size)
        self.linear_b = nn.Linear(input_size, self.embedding_size)
        self.linear_position = nn.Linear(2 * self.bins + 1, self.embedding_size)

    @classmethod
    def from_config(cls, config):
        return cls(
            embedding_size=config.getint("InputEmbedding", "pair_embedding_size"),
            position_bins=config.getint("InputEmbedding", "position_bins"),
        )

    def forward(self, target_feat, residue_index):
        a = self.linear_a(target_feat)
        b = self.linear_b(target_feat)
        z = a.unsqueeze(1) + b.unsqueeze(0)

        d = residue_index.reshape(-1, 1) - residue_index
        d = d.clamp(min=-self.bins, max=self.bins) + self.bins
        p = nn.functional.one_hot(d, num_classes=2 * self.bins + 1).float()
        p = self.linear_position(p)
        return z + p

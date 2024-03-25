from torch import nn

from nanofold.common.residue_definitions import RESIDUE_INDEX


class InputEmbedding(nn.Module):
    def __init__(self, embedding_size, position_bins):
        super().__init__()
        self.embedding_size = embedding_size
        self.bins = position_bins
        input_size = len(RESIDUE_INDEX)
        self.linear_a = nn.Linear(input_size, self.embedding_size)
        self.linear_b = nn.Linear(input_size, self.embedding_size)
        self.linear_position = nn.Linear(2 * self.bins + 1, self.embedding_size)

    def forward(self, target_feat, residue_index):
        a = self.linear_a(target_feat)
        b = self.linear_b(target_feat)
        z = a.unsqueeze(-2) + b.unsqueeze(-3)

        d = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        d = d.clamp(min=-self.bins, max=self.bins) + self.bins
        p = nn.functional.one_hot(d, num_classes=2 * self.bins + 1).float()
        p = self.linear_position(p)
        return z + p

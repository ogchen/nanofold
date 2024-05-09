from torch import nn


class RelativePositionEncoding(nn.Module):
    def __init__(self, num_bins, pair_embedding_size):
        super().__init__()
        self.bins = num_bins
        self.linear = nn.Linear(2 * self.bins + 1, pair_embedding_size)

    def forward(self, residue_index):
        d = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2) + self.bins
        d = d.clamp(min=0, max=2 * self.bins)
        rel_pos = nn.functional.one_hot(d.long(), num_classes=2 * self.bins + 1).float()
        return self.linear(rel_pos)

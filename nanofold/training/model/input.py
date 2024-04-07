from torch import nn

from nanofold.common.residue_definitions import RESIDUE_INDEX
from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA


class InputEmbedding(nn.Module):
    def __init__(self, pair_embedding_size, msa_embedding_size, position_bins):
        super().__init__()
        self.bins = position_bins
        target_input_size = len(RESIDUE_INDEX)
        msa_input_size = len(RESIDUE_INDEX_MSA) + 2
        self.linear_a = nn.Linear(target_input_size, pair_embedding_size)
        self.linear_b = nn.Linear(target_input_size, pair_embedding_size)
        self.linear_c = nn.Linear(target_input_size, msa_embedding_size)
        self.linear_position = nn.Linear(2 * self.bins + 1, pair_embedding_size)
        self.linear_msa = nn.Linear(msa_input_size, msa_embedding_size)

    def forward(self, target_feat, residue_index, msa_feat):
        a = self.linear_a(target_feat)
        b = self.linear_b(target_feat)
        z = a.unsqueeze(-2) + b.unsqueeze(-3)

        d = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        d = d.clamp(min=-self.bins, max=self.bins) + self.bins
        p = nn.functional.one_hot(d.long(), num_classes=2 * self.bins + 1).float()
        z = z + self.linear_position(p)

        m = self.linear_msa(msa_feat) + self.linear_c(target_feat.unsqueeze(-3))
        return m, z

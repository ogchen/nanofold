import torch.nn as nn
import torch.nn.functional as F

from nanofold.training.model.util import LinearWithView


class MSAPairWeightedAveraging(nn.Module):
    def __init__(
        self, msa_embedding_size, msa_averaging_embedding_size, pair_embedding_size, num_heads
    ):
        super().__init__()
        self.layer_norm_msa = nn.LayerNorm(msa_embedding_size)
        self.value = LinearWithView(msa_embedding_size, (num_heads, msa_averaging_embedding_size))
        self.bias = nn.Sequential(
            nn.LayerNorm(pair_embedding_size), nn.Linear(pair_embedding_size, num_heads)
        )
        self.gate = nn.Sequential(
            LinearWithView(msa_embedding_size, (num_heads, msa_averaging_embedding_size)),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(
            num_heads * msa_averaging_embedding_size, msa_embedding_size, bias=False
        )

    def forward(self, msa_rep, pair_rep):
        msa_rep = self.layer_norm_msa(msa_rep)
        v = self.value(msa_rep)
        b = self.bias(pair_rep)
        g = self.gate(msa_rep)
        w = F.softmax(b, dim=-2)
        o = g.transpose(-2, -3) * (w.movedim(-1, -3) @ v.movedim(-3, -2))
        msa_rep = self.proj(o.movedim(-3, -2).flatten(start_dim=-2))
        return msa_rep

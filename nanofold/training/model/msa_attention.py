from torch import nn
import math
import torch

from nanofold.training.model.util import LinearWithView


class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, pair_embedding_size, msa_embedding_size, num_heads, num_channels):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.layer_norm_msa = nn.LayerNorm(msa_embedding_size)
        self.layer_norm_pair = nn.LayerNorm(pair_embedding_size)
        self.linear_msa = LinearWithView(msa_embedding_size, (num_heads, num_channels))
        self.linear_pair = nn.Linear(pair_embedding_size, num_heads)
        self.projection = nn.Linear(num_heads * num_channels, msa_embedding_size)
        self.query = LinearWithView(msa_embedding_size, (num_heads, num_channels), bias=False)
        self.key = LinearWithView(msa_embedding_size, (num_heads, num_channels), bias=False)
        self.value = LinearWithView(msa_embedding_size, (num_heads, num_channels), bias=False)

    def forward(self, msa_rep, pair_rep):
        msa_rep = self.layer_norm_msa(msa_rep)
        q = self.query(msa_rep)
        k = self.key(msa_rep)
        v = self.value(msa_rep)
        b = self.linear_pair(self.layer_norm_pair(pair_rep))
        gates = torch.sigmoid(self.linear_msa(msa_rep))

        weights = (q.transpose(-2, -3) @ k.movedim(-3, -1)) / math.sqrt(self.num_channels)
        weights = weights + b.movedim(-1, -3).unsqueeze(-4)
        weights = nn.functional.softmax(weights, dim=-1)
        attention = gates * (weights @ v.transpose(-2, -3)).transpose(-2, -3)
        msa_rep = self.projection(
            attention.view((*attention.shape[:-2], self.num_channels * self.num_heads))
        )
        return msa_rep

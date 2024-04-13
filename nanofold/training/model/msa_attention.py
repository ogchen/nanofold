from torch import nn
import math

from nanofold.training.model.util import LinearWithView


class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, pair_embedding_size, msa_embedding_size, num_heads, num_channels):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.layer_norm = nn.LayerNorm(msa_embedding_size)
        self.gate = nn.Sequential(
            LinearWithView(msa_embedding_size, (num_heads, num_channels)),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(num_heads * num_channels, msa_embedding_size)
        self.query = LinearWithView(msa_embedding_size, (num_heads, num_channels), bias=False)
        self.key = LinearWithView(msa_embedding_size, (num_heads, num_channels), bias=False)
        self.value = LinearWithView(msa_embedding_size, (num_heads, num_channels), bias=False)
        if pair_embedding_size is not None:
            self.bias = nn.Sequential(
                nn.LayerNorm(pair_embedding_size),
                nn.Linear(pair_embedding_size, num_heads),
            )

    def forward(self, msa_rep, pair_rep=None):
        msa_rep = self.layer_norm(msa_rep)
        q = self.query(msa_rep)
        k = self.key(msa_rep)
        v = self.value(msa_rep)
        b = self.bias(pair_rep) if pair_rep is not None else None
        g = self.gate(msa_rep)

        weights = (q.transpose(-2, -3) @ k.movedim(-3, -1)) / math.sqrt(self.num_channels)
        if b is not None:
            weights = weights + b.movedim(-1, -3).unsqueeze(-4)
        weights = nn.functional.softmax(weights, dim=-1)
        attention = g * (weights @ v.transpose(-2, -3)).transpose(-2, -3)
        msa_rep = self.projection(attention.flatten(start_dim=-2))
        return msa_rep


class MSAColumnAttention(MSARowAttentionWithPairBias):
    def __init__(self, msa_embedding_size, num_heads, num_channels):
        super().__init__(None, msa_embedding_size, num_heads, num_channels)

    def forward(self, msa_rep):
        return super().forward(msa_rep.transpose(-2, -3)).transpose(-2, -3)


class MSAColumnGlobalAttention(nn.Module):
    def __init__(self, msa_embedding_size, num_heads, num_channels):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.layer_norm = nn.LayerNorm(msa_embedding_size)
        self.gate = nn.Sequential(
            LinearWithView(msa_embedding_size, (num_heads, num_channels)),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(num_heads * num_channels, msa_embedding_size)
        self.query = LinearWithView(msa_embedding_size, (num_heads, num_channels), bias=False)
        self.key = nn.Linear(msa_embedding_size, num_channels, bias=False)
        self.value = nn.Linear(msa_embedding_size, num_channels, bias=False)

    def forward(self, msa_rep):
        msa_rep = self.layer_norm(msa_rep)
        q = self.query(msa_rep).mean(dim=-4)
        k = self.key(msa_rep)
        v = self.value(msa_rep)
        g = self.gate(msa_rep)

        weights = (q @ k.movedim(-3, -1)) / math.sqrt(self.num_channels)
        weights = nn.functional.softmax(weights, dim=-1)
        attention = g * (weights @ v.transpose(-2, -3)).unsqueeze(-4)
        msa_rep = self.projection(attention.flatten(start_dim=-2))
        return msa_rep

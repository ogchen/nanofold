import math
import torch
from torch import nn

from nanofold.training.model.util import LinearWithView


class TriangleAttentionStartingNode(nn.Module):
    def __init__(self, pair_embedding_size, num_heads, num_channels):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.layer_norm = nn.LayerNorm(pair_embedding_size)
        self.query = LinearWithView(pair_embedding_size, (num_heads, num_channels), bias=False)
        self.key = LinearWithView(pair_embedding_size, (num_heads, num_channels), bias=False)
        self.value = LinearWithView(pair_embedding_size, (num_heads, num_channels), bias=False)
        self.bias = nn.Linear(pair_embedding_size, num_heads, bias=False)
        self.gate = LinearWithView(pair_embedding_size, (num_heads, num_channels), bias=False)
        self.out_proj = nn.Linear(num_channels * num_heads, pair_embedding_size)

    def forward(self, pair_rep):
        pair_rep = self.layer_norm(pair_rep)
        q = self.query(pair_rep).movedim(-2, -4)
        k = self.key(pair_rep).movedim(-2, -4)
        v = self.value(pair_rep).movedim(-2, -4)
        b = self.bias(pair_rep).movedim(-1, -3)
        g = torch.sigmoid(self.gate(pair_rep)).movedim(-2, -4)

        attention = nn.functional.softmax(
            q @ k.transpose(-1, -2) / math.sqrt(self.num_channels) + b.unsqueeze(-3), dim=-1
        )
        out = g * (attention @ v)
        return self.out_proj(out.movedim(-4, -1).flatten(start_dim=-2))


class TriangleAttentionEndingNode(nn.Module):
    def __init__(self, pair_embedding_size, num_heads, num_channels):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.layer_norm = nn.LayerNorm(pair_embedding_size)
        self.query = LinearWithView(pair_embedding_size, (num_heads, num_channels), bias=False)
        self.key = LinearWithView(pair_embedding_size, (num_heads, num_channels), bias=False)
        self.value = LinearWithView(pair_embedding_size, (num_heads, num_channels), bias=False)
        self.bias = nn.Linear(pair_embedding_size, num_heads, bias=False)
        self.gate = LinearWithView(pair_embedding_size, (num_heads, num_channels), bias=False)
        self.out_proj = nn.Linear(num_channels * num_heads, pair_embedding_size)

    def forward(self, pair_rep):
        pair_rep = self.layer_norm(pair_rep)
        q = self.query(pair_rep).movedim(-2, -4)
        k = self.key(pair_rep).movedim(-2, -4)
        v = self.value(pair_rep).movedim(-2, -4)
        b = self.bias(pair_rep).movedim(-1, -3)
        g = torch.sigmoid(self.gate(pair_rep)).movedim(-2, -4)

        attention = nn.functional.softmax(
            (q.movedim(-3, -2) @ k.movedim(-3, -1)).transpose(-2, -3) / math.sqrt(self.num_channels)
            + b.movedim(-2, -1).unsqueeze(-2),
            dim=-1,
        )
        out = g * (attention.transpose(-2, -3) @ v.transpose(-2, -3)).transpose(-2, -3)
        return self.out_proj(out.movedim(-4, -1).flatten(start_dim=-2))

import torch.nn as nn
import torch.nn.functional as F

from nanofold.train.model.util import LinearWithView


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
        self.out_proj = nn.Linear(num_channels * num_heads, pair_embedding_size, bias=False)

    def attention(self, q, k, v, b):
        return F.scaled_dot_product_attention(
            q.transpose(-3, -2),
            k.transpose(-3, -2),
            v.transpose(-3, -2),
            b.movedim(-1, -3).unsqueeze(-4),
        ).movedim(-3, -2)

    def forward(self, pair_rep):
        pair_rep = self.layer_norm(pair_rep)
        q = self.query(pair_rep)
        k = self.key(pair_rep)
        v = self.value(pair_rep)
        b = self.bias(pair_rep)
        g = self.gate(pair_rep)

        out = g * self.attention(q, k, v, b)
        return self.out_proj(out.flatten(start_dim=-2))


class TriangleAttentionEndingNode(TriangleAttentionStartingNode):
    def __init__(self, pair_embedding_size, num_heads, num_channels):
        super().__init__(pair_embedding_size, num_heads, num_channels)

    def attention(self, q, k, v, b):
        return F.scaled_dot_product_attention(
            q.movedim(-4, -2),
            k.movedim(-4, -2),
            v.movedim(-4, -2),
            b.movedim(-1, -3).transpose(-1, -2).unsqueeze(-4),
        ).movedim(-2, -4)

    def forward(self, pair_rep):
        return super().forward(pair_rep)

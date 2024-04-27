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

    def attention(self, q, k, v, b):
        weight = q.movedim(-3, -2) @ k.movedim(-3, -1) / (self.num_channels**2) + b.movedim(
            -1, -3
        ).unsqueeze(-4)
        attention = nn.functional.softmax(weight, dim=-1)
        return (attention @ v.transpose(-2, -3)).movedim(-3, -2)

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
        weight = q.movedim(-4, -2) @ k.movedim(-4, -1) / (self.num_channels**2) + b.movedim(
            -1, -3
        ).transpose(-1, -2).unsqueeze(-4)
        attention = nn.functional.softmax(weight, dim=-1)
        return (attention @ v.movedim(-4, -2)).transpose(-4, -2).movedim(-3, -2)

    def forward(self, pair_rep):
        return super().forward(pair_rep)

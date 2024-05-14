import torch.nn as nn
import torch.nn.functional as F

from nanofold.training.model.ada_ln import AdaLN
from nanofold.training.model.util import LinearWithView


class AttentionPairBias(nn.Module):
    def __init__(self, num_head, a_embedding_size, s_embedding_size, pair_embedding_size):
        super().__init__()
        self.embedding_size = a_embedding_size // num_head
        self.num_head = num_head
        self.ada_ln = AdaLN(a_embedding_size, s_embedding_size)
        self.layer_norm_a = nn.LayerNorm(a_embedding_size)
        self.query = LinearWithView(a_embedding_size, (num_head, self.embedding_size))
        self.key = LinearWithView(a_embedding_size, (num_head, self.embedding_size), bias=False)
        self.value = LinearWithView(a_embedding_size, (num_head, self.embedding_size), bias=False)
        self.bias = nn.Sequential(
            nn.LayerNorm(pair_embedding_size), nn.Linear(pair_embedding_size, num_head, bias=False)
        )
        self.gate = nn.Sequential(
            LinearWithView(a_embedding_size, (num_head, self.embedding_size), bias=False),
            nn.Sigmoid(),
        )
        self.projection_a = nn.Linear(num_head * self.embedding_size, a_embedding_size, bias=False)
        self.projection_out = nn.Sequential(
            nn.Linear(
                s_embedding_size,
                a_embedding_size,
            ),
            nn.Sigmoid(),
        )
        self.projection_out[0].bias.data.fill_(-2.0)

    def forward(self, a, s, pair_rep, beta):
        if s is None:
            a = self.layer_norm_a(a)
        else:
            a = self.ada_ln(a, s)

        q = self.query(a)
        k = self.key(a)
        v = self.value(a)
        b = self.bias(pair_rep) + beta.unsqueeze(-1) if beta is not None else self.bias(pair_rep)
        g = self.gate(a)

        attention = F.scaled_dot_product_attention(
            q.transpose(-3, -2),
            k.transpose(-3, -2),
            v.transpose(-3, -2),
            b.movedim(-1, -3),
        ) * g.transpose(-3, -2)
        a = self.projection_a(attention.transpose(-3, -2).flatten(start_dim=-2))

        if s is not None:
            a = self.projection_out(s) * a
        return a

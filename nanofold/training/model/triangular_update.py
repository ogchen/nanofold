import enum
import torch
from torch import nn


class UpdateDirection(enum.Enum):
    OUTGOING = 0
    INCOMING = 1


class TriangleMultiplicationUpdate(nn.Module):
    def __init__(self, pair_embedding_size, triangle_embedding_size, direction):
        super().__init__()
        self.layer_norm_pair = nn.LayerNorm(pair_embedding_size)
        self.layer_norm_update = nn.LayerNorm(triangle_embedding_size)
        self.linear_a1 = nn.Linear(pair_embedding_size, triangle_embedding_size)
        self.linear_a2 = nn.Linear(pair_embedding_size, triangle_embedding_size)
        self.linear_b1 = nn.Linear(pair_embedding_size, triangle_embedding_size)
        self.linear_b2 = nn.Linear(pair_embedding_size, triangle_embedding_size)
        self.linear_g = nn.Linear(pair_embedding_size, pair_embedding_size)
        self.linear_update = nn.Linear(triangle_embedding_size, pair_embedding_size)
        self.params = {
            UpdateDirection.OUTGOING: {"a_dim": -3, "b_dim": -4, "sum_dim": -2},
            UpdateDirection.INCOMING: {"a_dim": -2, "b_dim": -3, "sum_dim": -4},
        }[direction]

    def forward(self, pair_rep):
        pair_rep = self.layer_norm_pair(pair_rep)
        a = torch.sigmoid(self.linear_a1(pair_rep)) * self.linear_a2(pair_rep)
        b = torch.sigmoid(self.linear_b1(pair_rep)) * self.linear_b2(pair_rep)
        g = torch.sigmoid(self.linear_g(pair_rep))
        update = torch.sum(
            a.unsqueeze(self.params["a_dim"]) * b.unsqueeze(self.params["b_dim"]),
            dim=self.params["sum_dim"],
        )
        pair_rep = g * self.linear_update(self.layer_norm_update(update))
        return pair_rep

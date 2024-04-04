import enum
import torch
from torch import nn


class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, pair_embedding_size, num_channels):
        super().__init__()
        self.layer_norm_pair = nn.LayerNorm(pair_embedding_size)
        self.layer_norm_update = nn.LayerNorm(num_channels)
        self.gate_a = nn.Sequential(
            nn.Linear(pair_embedding_size, num_channels),
            nn.Sigmoid(),
        )
        self.gate_b = nn.Sequential(
            nn.Linear(pair_embedding_size, num_channels),
            nn.Sigmoid(),
        )
        self.linear_a = nn.Linear(pair_embedding_size, num_channels)
        self.linear_b = nn.Linear(pair_embedding_size, num_channels)
        self.linear_update = nn.Linear(num_channels, pair_embedding_size)
        self.gate = nn.Sequential(
            nn.Linear(pair_embedding_size, pair_embedding_size),
            nn.Sigmoid(),
        )

    def get_gate(self, pair_rep):
        return self.gate(pair_rep)

    def forward(self, pair_rep):
        pair_rep = self.layer_norm_pair(pair_rep)
        a = self.gate_a(pair_rep) * self.linear_a(pair_rep)
        b = self.gate_b(pair_rep) * self.linear_b(pair_rep)
        g = self.get_gate(pair_rep)
        update = torch.sum(
            a.unsqueeze(-3) * b.unsqueeze(-4),
            dim=-2,
        )
        pair_rep = g * self.linear_update(self.layer_norm_update(update))
        return pair_rep


class TriangleMultiplicationIncoming(TriangleMultiplicationOutgoing):
    def __init__(self, pair_embedding_size, num_channels):
        super().__init__(pair_embedding_size, num_channels)

    def get_gate(self, pair_rep_t):
        return self.gate(pair_rep_t.transpose(-2, -3))

    def forward(self, pair_rep):
        return super().forward(pair_rep.transpose(-2, -3))

import torch
import torch.nn as nn


class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, pair_embedding_size, num_channels):
        super().__init__()
        self.layer_norm_pair = nn.LayerNorm(pair_embedding_size)
        self.gate_a = nn.Sequential(
            nn.Linear(pair_embedding_size, num_channels, bias=False),
            nn.Sigmoid(),
        )
        self.gate_b = nn.Sequential(
            nn.Linear(pair_embedding_size, num_channels, bias=False),
            nn.Sigmoid(),
        )
        self.linear_a = nn.Linear(pair_embedding_size, num_channels, bias=False)
        self.linear_b = nn.Linear(pair_embedding_size, num_channels, bias=False)
        self.update_transition = nn.Sequential(
            nn.LayerNorm(num_channels), nn.Linear(num_channels, pair_embedding_size, bias=False)
        )
        self.gate = nn.Sequential(
            nn.Linear(pair_embedding_size, pair_embedding_size, bias=False),
            nn.Sigmoid(),
        )

    def update(self, a, b):
        return torch.einsum("...bikc,...bjkc->...bijc", a, b)

    def forward(self, pair_rep):
        pair_rep = self.layer_norm_pair(pair_rep)
        a = self.gate_a(pair_rep) * self.linear_a(pair_rep)
        b = self.gate_b(pair_rep) * self.linear_b(pair_rep)
        g = self.gate(pair_rep)
        update = self.update(a, b)
        pair_rep = g * self.update_transition(update)
        return pair_rep


class TriangleMultiplicationIncoming(TriangleMultiplicationOutgoing):
    def __init__(self, pair_embedding_size, num_channels):
        super().__init__(pair_embedding_size, num_channels)

    def update(self, a, b):
        return torch.einsum("...bkic,...bkjc->...bijc", a, b)

    def forward(self, pair_rep):
        return super().forward(pair_rep)

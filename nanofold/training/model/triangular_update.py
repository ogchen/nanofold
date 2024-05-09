from torch import nn


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
        return (
            (
                a.unsqueeze(-3).movedim(-2, -1).unsqueeze(-2)
                @ b.unsqueeze(-4).movedim(-2, -1).unsqueeze(-1)
            )
            .squeeze(-1)
            .squeeze(-1)
        )

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
        return (
            (
                a.movedim(-3, -1).unsqueeze(-3).unsqueeze(-2)
                @ b.movedim(-3, -1).unsqueeze(-4).unsqueeze(-1)
            )
            .squeeze(-1)
            .squeeze(-1)
        )

    def forward(self, pair_rep):
        return super().forward(pair_rep)

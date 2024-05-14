import torch.nn as nn
import torch.nn.functional as F


class Transition(nn.Module):
    def __init__(self, embedding_size, transition_multiplier):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_size)
        self.linear_a = nn.Linear(
            embedding_size, embedding_size * transition_multiplier, bias=False
        )
        self.linear_b = nn.Linear(
            embedding_size, embedding_size * transition_multiplier, bias=False
        )
        self.linear = nn.Linear(embedding_size * transition_multiplier, embedding_size, bias=False)

    def forward(self, x):
        x = self.layer_norm(x)
        a = self.linear_a(x)
        b = self.linear_b(x)
        x = self.linear(F.silu(a) * b)
        return x

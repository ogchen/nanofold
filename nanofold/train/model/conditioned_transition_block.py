import torch.nn as nn

from nanofold.train.model.ada_ln import AdaLN


class ConditionedTransitionBlock(nn.Module):
    def __init__(self, a_embedding_size, s_embedding_size, multiplier=2):
        super().__init__()
        self.ada_ln = AdaLN(a_embedding_size, s_embedding_size)
        self.linear_a = nn.Linear(a_embedding_size, a_embedding_size * multiplier, bias=False)
        self.linear_b = nn.Linear(a_embedding_size * multiplier, a_embedding_size, bias=False)
        self.transition = nn.Sequential(
            nn.Linear(a_embedding_size, a_embedding_size * multiplier, bias=False), nn.SiLU()
        )
        self.projection = nn.Sequential(
            nn.Linear(
                s_embedding_size,
                a_embedding_size,
            ),
            nn.Sigmoid(),
        )
        self.projection[0].bias.data.fill_(-2.0)

    def forward(self, a, s):
        a = self.ada_ln(a, s)
        b = self.transition(a) * self.linear_a(a)
        a = self.projection(s) * self.linear_b(b)
        return a

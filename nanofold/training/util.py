import math
import torch
from torch import nn


class LinearWithView(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, math.prod(out_features), *args, **kwargs)
        self.out_features = out_features

    def forward(self, x):
        out = self.linear(x)
        return out.view(*out.shape[:-1], *self.out_features)


def randint(*args):
    return torch.randint(*args, (1,)).item()


def accept_chain(chain):
    prob = 1 / 512 * max(min(len(chain), 512), 256)
    if torch.rand(1) < prob:
        return True


def crop_chain(chain, crop_size):
    if crop_size >= len(chain):
        return chain
    start = randint(0, len(chain) - crop_size + 1)
    return chain[start : start + crop_size]

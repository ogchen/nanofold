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


class DropoutByDimension(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, dim):
        shape = [d for d in x.shape]
        shape[dim] = 1
        return x * self.dropout(torch.ones(shape))

from torch import nn
import math


class LinearWithView(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, math.prod(out_features), *args, **kwargs)
        self.out_features = out_features

    def forward(self, x):
        out = self.linear(x)
        return out.view(*out.shape[:-1], *self.out_features)

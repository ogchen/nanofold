import torch.nn as nn


class AdaLN(nn.Module):
    def __init__(self, a_embedding_size, s_embedding_size):
        super().__init__()
        self.layer_norm_a = nn.LayerNorm(a_embedding_size, elementwise_affine=False, bias=False)
        self.layer_norm_s = nn.LayerNorm(s_embedding_size, bias=False)
        self.projection = nn.Sequential(nn.Linear(s_embedding_size, a_embedding_size), nn.Sigmoid())
        self.linear_s = nn.Linear(s_embedding_size, a_embedding_size, bias=False)

    def forward(self, a, s):
        a = self.layer_norm_a(a)
        s = self.layer_norm_s(s)
        a = self.projection(s) * a + self.linear_s(s)
        return a

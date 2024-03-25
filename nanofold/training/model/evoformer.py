from torch import nn
import torch


class Evoformer(nn.Module):
    def __init__(self, single_embedding_size, msa_embedding_size):
        super().__init__()
        self.linear_single = nn.Linear(msa_embedding_size, single_embedding_size)

    def forward(self, msa_rep, pair_rep):
        single_rep = self.linear_single(msa_rep[..., 0, :, :])
        return single_rep

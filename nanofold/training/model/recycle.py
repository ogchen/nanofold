import torch
from torch import nn


class RecyclingEmbedder(nn.Module):
    def __init__(self, pair_embedding_size, msa_embedding_size, device):
        super().__init__()
        self.bins = torch.cat(
            [torch.tensor([3.375], device=device), torch.arange(5.125, 22, 1.25, device=device)]
        )
        self.linear_one_hot = nn.Linear(len(self.bins), pair_embedding_size)
        self.layer_norm_pair = nn.LayerNorm(pair_embedding_size)
        self.layer_norm_msa = nn.LayerNorm(msa_embedding_size)

    def forward(self, msa_row, pair_rep, ca_coords):
        distance = torch.linalg.norm(ca_coords.unsqueeze(-2) - ca_coords.unsqueeze(-3), dim=-1)
        index = torch.argmin(torch.abs(distance.unsqueeze(-1) - self.bins), dim=-1)
        distance = self.linear_one_hot(
            torch.nn.functional.one_hot(index, num_classes=len(self.bins)).float()
        )
        pair_rep = distance + self.layer_norm_pair(pair_rep)
        msa_row = self.layer_norm_msa(msa_row)
        return msa_row, pair_rep

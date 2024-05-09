from torch import nn
import torch


class OuterProductMean(nn.Module):
    def __init__(self, pair_embedding_size, msa_embedding_size, product_embedding_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(msa_embedding_size)
        self.linear_a = nn.Linear(msa_embedding_size, product_embedding_size, bias=False)
        self.linear_b = nn.Linear(msa_embedding_size, product_embedding_size, bias=False)
        self.projection = nn.Linear(
            product_embedding_size * product_embedding_size, pair_embedding_size
        )

    def forward(self, msa_rep):
        msa_rep = self.layer_norm(msa_rep)
        a = self.linear_a(msa_rep)
        b = self.linear_b(msa_rep)
        outer = a.unsqueeze(-2).unsqueeze(-1) @ b.unsqueeze(-3).unsqueeze(-2)
        outer = torch.mean(outer, dim=-5).flatten(start_dim=-2)
        return self.projection(outer)

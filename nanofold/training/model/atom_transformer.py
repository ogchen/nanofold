import torch
import torch.nn as nn

from nanofold.training.model.diffusion_transformer import DiffusionTransformer


class AtomTransformer(nn.Module):
    def __init__(
        self,
        q_embedding_size,
        c_embedding_size,
        p_embedding_size,
        num_block,
        num_head,
        num_queries,
        num_keys,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_keys = num_keys
        self.diffusion_transformer = DiffusionTransformer(
            q_embedding_size, c_embedding_size, p_embedding_size, num_block, num_head
        )

    def forward(self, q, c, pair_rep):
        device = pair_rep.device
        centres = torch.arange(
            (self.num_queries - 1) / 2, q.size(-2), self.num_queries, device=device
        )
        row_mask = (
            torch.abs(torch.arange(pair_rep.size(-3), device=device).unsqueeze(-1) - centres)
            < self.num_queries / 2
        )
        col_mask = (
            torch.abs(torch.arange(pair_rep.size(-2), device=device).unsqueeze(-1) - centres)
            < self.num_keys / 2
        )
        mask = torch.any(row_mask.unsqueeze(-2) & col_mask.unsqueeze(-3), dim=-1)
        beta = torch.ones_like(mask) * -(10**10) * ~mask
        return self.diffusion_transformer(q, c, pair_rep, beta)

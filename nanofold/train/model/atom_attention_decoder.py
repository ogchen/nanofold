import torch.nn as nn
import torch

from nanofold.train.model.atom_transformer import AtomTransformer


class AtomAttentionDecoder(nn.Module):
    def __init__(
        self,
        q_embedding_size,
        c_embedding_size,
        p_embedding_size,
        a_embedding_size,
        num_block,
        num_head,
        num_queries,
        num_keys,
    ):
        super().__init__()
        self.atom_transformer = AtomTransformer(
            q_embedding_size,
            c_embedding_size,
            p_embedding_size,
            num_block,
            num_head,
            num_queries,
            num_keys,
        )
        self.linear = nn.Linear(a_embedding_size, q_embedding_size, bias=False)
        self.positions_update = nn.Sequential(
            nn.LayerNorm(q_embedding_size), nn.Linear(q_embedding_size, 3)
        )

    def forward(self, a, q, c, p):
        q = self.linear(torch.tile(a, (3, 1))) + q
        q = self.atom_transformer(q, c, p)
        return self.positions_update(q)

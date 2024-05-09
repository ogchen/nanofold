import torch
import torch.nn as nn

from nanofold.training.model.atom_attention_encoder import AtomAttentionEncoder


class InputEmbedding(nn.Module):
    def __init__(
        self,
        atom_embedding_size,
        atom_pair_embedding_size,
        token_embedding_size,
        num_block,
        num_head,
        num_queries,
        num_keys,
    ):
        super().__init__()
        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_embedding_size,
            atom_pair_embedding_size,
            token_embedding_size,
            num_block,
            num_head,
            num_queries,
            num_keys,
        )

    def forward(self, batch):
        ref_pos = batch["ref_pos"]
        ref_space_uid = batch["ref_space_uid"]
        a, _, _, _ = self.atom_attention_encoder(ref_pos, ref_space_uid, None, None, None)
        return torch.concat(
            [a, batch["restype"], batch["profile"], batch["deletion_mean"].unsqueeze(-1)], dim=-1
        )

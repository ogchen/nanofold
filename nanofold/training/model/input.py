import torch
import torch.nn as nn

from nanofold.common.residue_definitions import RESIDUE_INDEX
from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA
from nanofold.training.model.atom_attention_encoder import AtomAttentionEncoder


class InputEmbedding(nn.Module):
    def __init__(
        self,
        atom_embedding_size,
        atom_pair_embedding_size,
        token_embedding_size,
        single_embedding_size,
        pair_embedding_size,
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
            single_embedding_size,
            pair_embedding_size,
            num_block,
            num_head,
            num_queries,
            num_keys,
        )
        self.embedding_size = token_embedding_size + len(RESIDUE_INDEX) + len(RESIDUE_INDEX_MSA) + 1

    def forward(self, batch):
        ref_pos = batch["ref_pos"]
        ref_space_uid = batch["ref_space_uid"]
        a, _, _, _ = self.atom_attention_encoder(ref_pos, ref_space_uid, None, None, None)
        return torch.concat(
            [a, batch["restype"], batch["profile"], batch["deletion_mean"].unsqueeze(-1)], dim=-1
        )

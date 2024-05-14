import torch.nn as nn

from nanofold.training.model.input import InputEmbedding
from nanofold.training.model.relative_position_encoding import RelativePositionEncoding


class NanofoldInput(nn.Module):
    def __init__(
        self,
        single_embedding_size,
        pair_embedding_size,
        input_atom_embedding_size,
        input_atom_pair_embedding_size,
        input_token_embedding_size,
        position_bins,
        num_atom_transformer_blocks,
        num_atom_transformer_heads,
        num_atom_transformer_queries,
        num_atom_transformer_keys,
    ):
        super().__init__()

        self.input_embedder = InputEmbedding(
            input_atom_embedding_size,
            input_atom_pair_embedding_size,
            input_token_embedding_size,
            single_embedding_size,
            pair_embedding_size,
            num_atom_transformer_blocks,
            num_atom_transformer_heads,
            num_atom_transformer_queries,
            num_atom_transformer_keys,
        )
        input_embedding_size = self.input_embedder.embedding_size
        self.linear_input_single = nn.Linear(input_embedding_size, single_embedding_size)
        self.linear_input_pair_a = nn.Linear(input_embedding_size, pair_embedding_size)
        self.linear_input_pair_b = nn.Linear(input_embedding_size, pair_embedding_size)
        self.relative_position_encoding = RelativePositionEncoding(
            position_bins, pair_embedding_size
        )

    def forward(self, features):
        input = self.input_embedder(features)
        single_rep_init = self.linear_input_single(input)
        pair_rep_init = self.linear_input_pair_a(input.unsqueeze(-2)) + self.linear_input_pair_b(
            input.unsqueeze(-3)
        )
        pair_rep_init = pair_rep_init + self.relative_position_encoding(features["residue_index"])
        return input, single_rep_init, pair_rep_init

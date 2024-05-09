import torch.nn as nn

from nanofold.training.model.attention_pair_bias import AttentionPairBias
from nanofold.training.model.conditioned_transition_block import ConditionedTransitionBlock


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, atom_embedding_size, atom_pair_embedding_size, num_head):

        super().__init__()
        self.attention_pair_bias = AttentionPairBias(
            num_head, atom_embedding_size, atom_embedding_size, atom_pair_embedding_size
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            atom_embedding_size, atom_embedding_size
        )

    def forward(self, a, s, pair_rep, beta):
        b = self.attention_pair_bias(a, s, pair_rep, beta)
        a = b + self.conditioned_transition_block(a, s)
        return a


class DiffusionTransformer(nn.Module):
    def __init__(self, atom_embedding_size, atom_pair_embedding_size, num_block, num_head):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(atom_embedding_size, atom_pair_embedding_size, num_head)
                for _ in range(num_block)
            ]
        )

    def forward(self, a, s, pair_rep, beta):
        for block in self.blocks:
            a = block(a, s, pair_rep, beta)
        return a

import torch.nn as nn

from nanofold.train.model.attention_pair_bias import AttentionPairBias
from nanofold.train.model.conditioned_transition_block import ConditionedTransitionBlock


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, a_embedding_size, s_embedding_size, pair_embedding_size, num_head):

        super().__init__()
        self.attention_pair_bias = AttentionPairBias(
            num_head, a_embedding_size, s_embedding_size, pair_embedding_size
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            a_embedding_size, s_embedding_size
        )

    def forward(self, a, s, pair_rep, beta):
        b = self.attention_pair_bias(a, s, pair_rep, beta)
        a = b + self.conditioned_transition_block(a, s)
        return a


class DiffusionTransformer(nn.Module):
    def __init__(
        self, a_embedding_size, s_embedding_size, pair_embedding_size, num_block, num_head
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    a_embedding_size, s_embedding_size, pair_embedding_size, num_head
                )
                for _ in range(num_block)
            ]
        )

    def forward(self, a, s, pair_rep, beta):
        for block in self.blocks:
            a = block(a, s, pair_rep, beta)
        return a

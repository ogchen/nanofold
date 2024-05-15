import torch
import torch.nn as nn

from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA
from nanofold.train.model.msa_averaging import MSAPairWeightedAveraging
from nanofold.train.model.outer_product_mean import OuterProductMean
from nanofold.train.model.transition import Transition
from nanofold.train.model.triangular_attention import TriangleAttentionStartingNode
from nanofold.train.model.triangular_attention import TriangleAttentionEndingNode
from nanofold.train.model.triangular_update import TriangleMultiplicationOutgoing
from nanofold.train.model.triangular_update import TriangleMultiplicationIncoming
from nanofold.train.model.util import DropoutByDimension


class MSAModuleBlock(nn.Module):
    def __init__(
        self,
        pair_embedding_size,
        msa_embedding_size,
        msa_averaging_embedding_size,
        product_embedding_size,
        num_triangular_update_channels,
        num_triangular_attention_channels,
        num_msa_heads,
        num_pair_heads,
        transition_multiplier,
        p_msa_dropout=0.15,
        p_pair_dropout=0.25,
    ):
        super().__init__()
        self.msa_dropout = DropoutByDimension(p_msa_dropout)
        self.pair_dropout = DropoutByDimension(p_pair_dropout)
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
            msa_embedding_size, msa_averaging_embedding_size, pair_embedding_size, num_msa_heads
        )
        self.msa_transition = Transition(msa_embedding_size, transition_multiplier)
        self.outer_product_mean = OuterProductMean(
            pair_embedding_size, msa_embedding_size, product_embedding_size
        )
        self.triangle_update_outgoing = TriangleMultiplicationOutgoing(
            pair_embedding_size, num_triangular_update_channels
        )
        self.triangle_update_incoming = TriangleMultiplicationIncoming(
            pair_embedding_size, num_triangular_update_channels
        )
        self.triangle_attention_starting = TriangleAttentionStartingNode(
            pair_embedding_size, num_pair_heads, num_triangular_attention_channels
        )
        self.triangle_attention_ending = TriangleAttentionEndingNode(
            pair_embedding_size, num_pair_heads, num_triangular_attention_channels
        )
        self.pair_transition = Transition(pair_embedding_size, transition_multiplier)

    def forward(self, msa_rep, pair_rep):
        pair_rep = pair_rep + self.outer_product_mean(msa_rep)

        msa_rep = msa_rep + self.msa_dropout(
            self.msa_pair_weighted_averaging(msa_rep, pair_rep), dim=-3
        )
        msa_rep = msa_rep + self.msa_transition(msa_rep)

        pair_rep = pair_rep + self.pair_dropout(self.triangle_update_outgoing(pair_rep), dim=-3)
        pair_rep = pair_rep + self.pair_dropout(self.triangle_update_incoming(pair_rep), dim=-3)
        pair_rep = pair_rep + self.pair_dropout(self.triangle_attention_starting(pair_rep), dim=-3)
        pair_rep = pair_rep + self.pair_dropout(self.triangle_attention_ending(pair_rep), dim=-2)
        pair_rep = pair_rep + self.pair_transition(pair_rep)
        return msa_rep, pair_rep


class MSAModule(nn.Module):
    def __init__(
        self,
        num_block,
        input_embedding_size,
        msa_embedding_size,
        num_msa_samples,
        pair_embedding_size,
        msa_averaging_embedding_size,
        product_embedding_size,
        num_triangular_update_channels,
        num_triangular_attention_channels,
        num_msa_heads,
        num_pair_heads,
        transition_multiplier,
    ):
        super().__init__()
        self.num_msa_samples = num_msa_samples
        self.linear_msa = nn.Linear(len(RESIDUE_INDEX_MSA) + 2, msa_embedding_size)
        self.linear_input = nn.Linear(input_embedding_size, msa_embedding_size)
        self.blocks = nn.ModuleList(
            [
                MSAModuleBlock(
                    pair_embedding_size,
                    msa_embedding_size,
                    msa_averaging_embedding_size,
                    product_embedding_size,
                    num_triangular_update_channels,
                    num_triangular_attention_channels,
                    num_msa_heads,
                    num_pair_heads,
                    transition_multiplier,
                )
                for _ in range(num_block)
            ]
        )

    def forward(self, features, pair_rep, input):
        msa = features["msa"]
        has_deletion = features["has_deletion"]
        deletion_value = features["deletion_value"]
        msa = torch.concat([msa, has_deletion, deletion_value], dim=-1)
        index = torch.randperm(msa.size(-3))[: self.num_msa_samples]
        msa_rep = self.linear_msa(msa[index].to(input.device))
        msa_rep = msa_rep + self.linear_input(input)

        for block in self.blocks:
            msa_rep, pair_rep = block(msa_rep, pair_rep)
        return pair_rep

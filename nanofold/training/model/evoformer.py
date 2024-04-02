from torch import nn
import torch

from nanofold.training.model.msa_attention import MSAColumnAttention
from nanofold.training.model.msa_attention import MSARowAttentionWithPairBias


class EvoformerBlock(nn.Module):
    def __init__(
        self,
        pair_embedding_size,
        msa_embedding_size,
        num_heads,
        num_channels,
        p_msa_dropout=0.15,
        p_pair_dropout=0.25,
    ):
        super().__init__()
        self.msa_row_attention = MSARowAttentionWithPairBias(
            pair_embedding_size, msa_embedding_size, num_heads, num_channels
        )
        self.msa_col_attention = MSAColumnAttention(msa_embedding_size, num_heads, num_channels)
        self.msa_dropout = nn.Dropout(p=p_msa_dropout)
        self.pair_dropout = nn.Dropout(p=p_pair_dropout)

    def forward(self, msa_rep, pair_rep):
        row_attention = self.msa_row_attention(msa_rep, pair_rep)
        row_attention = row_attention * self.msa_dropout(
            torch.ones_like(row_attention[..., :1, :, :])
        )
        msa_rep = msa_rep + row_attention
        msa_rep = msa_rep + self.msa_col_attention(msa_rep)

        return msa_rep, pair_rep


class Evoformer(nn.Module):
    def __init__(
        self,
        single_embedding_size,
        pair_embedding_size,
        msa_embedding_size,
        num_blocks,
        num_heads,
        num_channels,
    ):
        super().__init__()
        self.linear_single = nn.Linear(msa_embedding_size, single_embedding_size)
        self.blocks = nn.ModuleList(
            [
                EvoformerBlock(pair_embedding_size, msa_embedding_size, num_heads, num_channels)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, msa_rep, pair_rep):
        for block in self.blocks:
            msa_rep, pair_rep = block(msa_rep, pair_rep)

        single_rep = self.linear_single(msa_rep[..., 0, :, :])
        return single_rep

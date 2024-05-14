import torch.nn as nn

from nanofold.training.model.msa_module import MSAModule
from nanofold.training.model.pairformer import Pairformer
from nanofold.training.model.template_embedder import TemplateEmbedder


class NanofoldTrunk(nn.Module):
    def __init__(
        self,
        single_embedding_size,
        pair_embedding_size,
        input_embedding_size,
        product_embedding_size,
        num_msa_samples,
        num_msa_blocks,
        msa_embedding_size,
        msa_averaging_embedding_size,
        num_msa_heads,
        msa_transition_multiplier,
        num_triangular_update_channels,
        num_triangular_attention_channels,
        num_triangular_attention_heads,
        num_template_blocks,
        template_embedding_size,
        num_pairformer_blocks,
        num_pair_heads,
        pairformer_transition_multiplier,
    ):
        super().__init__()
        self.transition_pair = nn.Sequential(
            nn.LayerNorm(pair_embedding_size),
            nn.Linear(pair_embedding_size, pair_embedding_size, bias=False),
        )
        self.template_embedder = TemplateEmbedder(
            template_embedding_size,
            pair_embedding_size,
            num_triangular_update_channels,
            num_triangular_attention_channels,
            num_triangular_attention_heads,
            num_pair_heads,
            pairformer_transition_multiplier,
            num_template_blocks,
        )
        self.msa_module = MSAModule(
            num_msa_blocks,
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
            msa_transition_multiplier,
        )
        self.transition_single = nn.Sequential(
            nn.LayerNorm(single_embedding_size),
            nn.Linear(single_embedding_size, single_embedding_size, bias=False),
        )
        self.pairformer = Pairformer(
            single_embedding_size,
            pair_embedding_size,
            num_triangular_update_channels,
            num_triangular_attention_channels,
            num_triangular_attention_heads,
            num_pair_heads,
            pairformer_transition_multiplier,
            num_pairformer_blocks,
        )

    def forward(self, features, input, pair_rep_init, single_rep_init, pair_rep, single_rep):
        pair_rep = pair_rep_init + self.transition_pair(pair_rep)
        pair_rep = pair_rep + self.template_embedder(features, pair_rep)
        pair_rep = pair_rep + self.msa_module(features, pair_rep, input)
        single_rep = single_rep_init + self.transition_single(single_rep)
        single_rep, pair_rep = self.pairformer(single_rep, pair_rep)
        return single_rep, pair_rep

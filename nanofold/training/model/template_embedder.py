import torch
import torch.nn as nn

from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA
from nanofold.training.model.pairformer import Pairformer


class TemplateEmbedder(nn.Module):
    def __init__(
        self,
        template_embedding_size,
        pair_embedding_size,
        num_triangular_update_channels,
        num_triangular_attention_channels,
        num_triangular_attention_heads,
        num_pair_heads,
        transition_multiplier,
        num_blocks,
    ):
        super().__init__()
        self.pair_embedder = nn.Sequential(
            nn.LayerNorm(pair_embedding_size),
            nn.Linear(pair_embedding_size, template_embedding_size, bias=False),
        )
        template_input_size = 39 + 1 + 3 + 2 * len(RESIDUE_INDEX_MSA)
        self.linear = nn.Linear(template_input_size, template_embedding_size)
        self.layer_norm = nn.LayerNorm(template_embedding_size)
        self.transition = nn.Sequential(
            nn.ReLU(), nn.Linear(template_embedding_size, template_embedding_size, bias=False)
        )
        self.pairformer_stack = Pairformer(
            0,
            pair_embedding_size,
            num_triangular_update_channels,
            num_triangular_attention_channels,
            num_triangular_attention_heads,
            num_pair_heads,
            transition_multiplier,
            num_blocks,
        )

    def forward(self, features, pair_rep):
        template_backbone_frame_mask = features["template_backbone_frame_mask"]
        template_distogram = features["template_distogram"]
        template_unit_vector = features["template_unit_vector"]
        template_restype = features["template_restype"]

        s = template_restype.shape
        num_res = s[-2]
        s = [*s[:-2], num_res, num_res, s[-1]]

        b = template_backbone_frame_mask.unsqueeze(-1) & template_backbone_frame_mask.unsqueeze(-2)
        a = torch.concat(
            [
                template_distogram,
                b.unsqueeze(-1),
                template_unit_vector,
                torch.tile(template_restype, (1, num_res)).view(s),
                torch.tile(template_restype, (num_res, 1)).view(s),
            ],
            dim=-1,
        )

        v = self.pair_embedder(pair_rep.unsqueeze(-4)) + self.linear(a)
        _, v = self.pairformer_stack(None, v)
        u = self.layer_norm(v).mean(dim=-4)
        u = self.transition(u)
        return u

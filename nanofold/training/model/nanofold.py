import numpy as np
import torch
from torch import nn

from nanofold.training.frame import Frame
from nanofold.training.model.evoformer import Evoformer
from nanofold.training.model.input import InputEmbedding
from nanofold.training.model.recycle import RecyclingEmbedder
from nanofold.training.model.structure import StructureModule


class Nanofold(nn.Module):
    def __init__(
        self,
        num_recycle,
        num_structure_layers,
        single_embedding_size,
        pair_embedding_size,
        msa_embedding_size,
        num_triangular_update_channels,
        num_triangular_attention_channels,
        product_embedding_size,
        position_bins,
        num_evoformer_blocks,
        num_evoformer_msa_heads,
        num_evoformer_pair_heads,
        num_evoformer_channels,
        evoformer_transition_multiplier,
        dropout,
        ipa_embedding_size,
        num_query_points,
        num_value_points,
        num_heads,
    ):
        super().__init__()
        self.num_recycle = num_recycle
        self.msa_embedding_size = msa_embedding_size
        self.pair_embedding_size = pair_embedding_size
        self.input_embedder = InputEmbedding(pair_embedding_size, msa_embedding_size, position_bins)
        self.recycling_embedder = RecyclingEmbedder(pair_embedding_size, msa_embedding_size)
        self.evoformer = Evoformer(
            single_embedding_size,
            pair_embedding_size,
            msa_embedding_size,
            num_triangular_update_channels,
            num_triangular_attention_channels,
            product_embedding_size,
            num_evoformer_blocks,
            num_evoformer_msa_heads,
            num_evoformer_pair_heads,
            num_evoformer_channels,
            evoformer_transition_multiplier,
        )
        self.structure_module = StructureModule(
            num_structure_layers,
            single_embedding_size,
            pair_embedding_size,
            dropout,
            ipa_embedding_size,
            num_query_points,
            num_value_points,
            num_heads,
        )
        self.recycling_embedder = RecyclingEmbedder(pair_embedding_size, msa_embedding_size)

    @staticmethod
    def get_args(config):
        return {
            "num_recycle": config.getint("General", "num_recycle"),
            "num_structure_layers": config.getint("StructureModule", "num_layers"),
            "single_embedding_size": config.getint("General", "single_embedding_size"),
            "pair_embedding_size": config.getint("General", "pair_embedding_size"),
            "msa_embedding_size": config.getint("General", "msa_embedding_size"),
            "position_bins": config.getint("General", "position_bins"),
            "num_triangular_update_channels": config.getint(
                "Evoformer", "num_triangular_update_channels"
            ),
            "num_triangular_attention_channels": config.getint(
                "Evoformer", "num_triangular_attention_channels"
            ),
            "product_embedding_size": config.getint("Evoformer", "product_embedding_size"),
            "num_evoformer_blocks": config.getint("Evoformer", "num_blocks"),
            "num_evoformer_msa_heads": config.getint("Evoformer", "num_msa_heads"),
            "num_evoformer_pair_heads": config.getint("Evoformer", "num_pair_heads"),
            "num_evoformer_channels": config.getint("Evoformer", "num_channels"),
            "evoformer_transition_multiplier": config.getint("Evoformer", "transition_multiplier"),
            "dropout": config.getfloat("StructureModule", "dropout"),
            "ipa_embedding_size": config.getint("InvariantPointAttention", "embedding_size"),
            "num_query_points": config.getint("InvariantPointAttention", "num_query_points"),
            "num_value_points": config.getint("InvariantPointAttention", "num_value_points"),
            "num_heads": config.getint("InvariantPointAttention", "num_heads"),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**cls.get_args(config))

    def forward(self, batch):
        num_recycle = np.random.randint(self.num_recycle) + 1 if self.training else self.num_recycle
        s = batch["positions"].shape
        prev_msa_row = torch.zeros((*s, self.msa_embedding_size))
        prev_ca_coords = torch.zeros((*s, 3))
        prev_pair_rep = torch.zeros((*s, s[-1], self.pair_embedding_size))

        for i in range(num_recycle):
            prev_msa_row = prev_msa_row.detach()
            prev_ca_coords = prev_ca_coords.detach()
            prev_pair_rep = prev_pair_rep.detach()

            msa_rep, pair_rep = self.input_embedder(
                batch["target_feat"], batch["positions"], batch["msa_feat"]
            )
            msa_row_update, pair_rep_update = self.recycling_embedder(
                prev_msa_row, prev_pair_rep, prev_ca_coords
            )
            msa_rep[..., 0, :, :] = msa_rep[..., 0, :, :] + msa_row_update
            pair_rep = pair_rep + pair_rep_update

            msa_rep, pair_rep, single_rep = self.evoformer(msa_rep, pair_rep)

            coords, fape_loss, aux_loss = self.structure_module(
                single_rep,
                pair_rep,
                batch["local_coords"],
                (
                    Frame(
                        rotations=batch["rotations"],
                        translations=batch["translations"],
                    )
                    if i == num_recycle - 1
                    else None
                ),
            )
            prev_msa_row = msa_rep[..., 0, :, :]
            prev_pair_rep = pair_rep
            prev_ca_coords = coords[..., 1, :]
        return coords, fape_loss + aux_loss

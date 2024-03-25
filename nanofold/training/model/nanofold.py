import torch
from torch import nn

from nanofold.training.frame import Frame
from nanofold.training.model.evoformer import Evoformer
from nanofold.training.model.input import InputEmbedding
from nanofold.training.model.structure import StructureModule


class Nanofold(nn.Module):
    def __init__(
        self,
        num_layers,
        single_embedding_size,
        pair_embedding_size,
        msa_embedding_size,
        position_bins,
        dropout,
        ipa_embedding_size,
        num_query_points,
        num_value_points,
        num_heads,
    ):
        super().__init__()
        self.input_embedder = InputEmbedding(pair_embedding_size, msa_embedding_size, position_bins)
        self.evoformer = Evoformer(single_embedding_size, msa_embedding_size)
        self.structure_module = StructureModule(
            num_layers,
            single_embedding_size,
            pair_embedding_size,
            dropout,
            ipa_embedding_size,
            num_query_points,
            num_value_points,
            num_heads,
        )

    @staticmethod
    def get_args(config):
        return {
            "num_layers": config.getint("StructureModule", "num_layers"),
            "single_embedding_size": config.getint("General", "single_embedding_size"),
            "pair_embedding_size": config.getint("General", "pair_embedding_size"),
            "msa_embedding_size": config.getint("General", "msa_embedding_size"),
            "position_bins": config.getint("General", "position_bins"),
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
        msa_rep, pair_rep = self.input_embedder(
            batch["target_feat"], batch["positions"], batch["msa_feat"]
        )
        single_rep = self.evoformer(msa_rep, pair_rep)
        coords, fape_loss, aux_loss = self.structure_module(
            single_rep,
            pair_rep,
            batch["local_coords"],
            Frame(
                rotations=batch["rotations"],
                translations=batch["translations"],
            ),
        )
        return coords, fape_loss + aux_loss

import torch
from torch import nn

from nanofold.training.frame import Frame
from nanofold.training.loss import DistogramLoss
from nanofold.training.model.evoformer import Evoformer
from nanofold.training.model.input import InputEmbedding
from nanofold.training.model.masked_msa import MaskedMSAPredictor
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
        structure_dropout,
        ipa_embedding_size,
        num_ipa_query_points,
        num_ipa_value_points,
        num_ipa_heads,
        num_distogram_bins,
        num_distogram_channels,
        num_lddt_bins,
        num_lddt_channels,
        use_grad_checkpoint,
        device,
    ):
        super().__init__()
        self.device = device
        self.num_recycle = num_recycle
        self.msa_embedding_size = msa_embedding_size
        self.pair_embedding_size = pair_embedding_size
        self.input_embedder = InputEmbedding(pair_embedding_size, msa_embedding_size, position_bins)
        self.recycling_embedder = RecyclingEmbedder(pair_embedding_size, msa_embedding_size, device)
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
            device,
        )
        self.structure_module = StructureModule(
            num_structure_layers,
            single_embedding_size,
            pair_embedding_size,
            structure_dropout,
            ipa_embedding_size,
            num_ipa_query_points,
            num_ipa_value_points,
            num_ipa_heads,
            num_lddt_bins,
            num_lddt_channels,
            device,
        )
        self.distogram_loss = DistogramLoss(
            pair_embedding_size, num_distogram_bins, num_distogram_channels, device
        )
        self.msa_predictor = MaskedMSAPredictor(msa_embedding_size)
        self.use_grad_checkpoint = use_grad_checkpoint

    @staticmethod
    def get_args(config):
        return {
            "num_recycle": config["num_recycle"],
            "num_structure_layers": config["num_structure_layers"],
            "single_embedding_size": config["single_embedding_size"],
            "pair_embedding_size": config["pair_embedding_size"],
            "msa_embedding_size": config["msa_embedding_size"],
            "position_bins": config["position_bins"],
            "num_triangular_update_channels": config["num_triangular_update_channels"],
            "num_triangular_attention_channels": config["num_triangular_attention_channels"],
            "product_embedding_size": config["product_embedding_size"],
            "num_evoformer_blocks": config["num_evoformer_blocks"],
            "num_evoformer_msa_heads": config["num_evoformer_msa_heads"],
            "num_evoformer_pair_heads": config["num_evoformer_pair_heads"],
            "num_evoformer_channels": config["num_evoformer_channels"],
            "evoformer_transition_multiplier": config["evoformer_transition_multiplier"],
            "structure_dropout": config["structure_dropout"],
            "ipa_embedding_size": config["ipa_embedding_size"],
            "num_ipa_query_points": config["num_ipa_query_points"],
            "num_ipa_value_points": config["num_ipa_value_points"],
            "num_ipa_heads": config["num_ipa_heads"],
            "num_distogram_bins": config["num_distogram_bins"],
            "num_distogram_channels": config["num_distogram_channels"],
            "num_lddt_bins": config["num_lddt_bins"],
            "num_lddt_channels": config["num_lddt_channels"],
            "use_grad_checkpoint": config["use_grad_checkpoint"],
            "device": config["device"],
        }

    @classmethod
    def from_config(cls, config):
        return cls(**cls.get_args(config))

    def run_evoformer(self, *args):
        if self.use_grad_checkpoint or not self.training:
            return torch.utils.checkpoint.checkpoint(
                lambda *inputs: self.evoformer(*inputs), *args, use_reentrant=False
            )
        return self.evoformer(*args)

    def forward(self, batch):
        num_recycle = (
            torch.randint(self.num_recycle, (1,)) + 1 if self.training else self.num_recycle
        )
        fape_clamp = 10.0 if torch.rand(1) < 0.9 and self.training else None

        s = batch["positions"].shape
        prev_msa_row = torch.zeros((*s, self.msa_embedding_size), device=self.device)
        prev_ca_coords = torch.zeros((*s, 3), device=self.device)
        prev_pair_rep = torch.zeros((*s, s[-1], self.pair_embedding_size), device=self.device)

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

            msa_rep, pair_rep, single_rep = self.run_evoformer(msa_rep, pair_rep)

            coords, chain_plddt, chain_lddt, fape_loss, conf_loss, aux_loss = self.structure_module(
                single_rep,
                pair_rep,
                batch["local_coords"],
                (
                    Frame(
                        rotations=batch["rotations"],
                        translations=batch["translations"],
                    )
                    if i == num_recycle - 1 and "translations" in batch
                    else None
                ),
                fape_clamp,
            )
            prev_msa_row = msa_rep[..., 0, :, :]
            prev_pair_rep = pair_rep
            prev_ca_coords = coords[..., 1, :]

        msa_loss = self.msa_predictor(msa_rep, batch["msa_mask"], batch.get("masked_msa_truth"))

        dist_loss = (
            self.distogram_loss(pair_rep, batch["translations"])
            if "translations" in batch
            else None
        )

        return coords, chain_plddt, chain_lddt, fape_loss, conf_loss, aux_loss, dist_loss, msa_loss

import torch
from torch import nn

from nanofold.training.loss import compute_fape_loss
from nanofold.training.loss import LDDTPredictor
from nanofold.training.frame import Frame
from nanofold.training.model.backbone_update import BackboneUpdate
from nanofold.training.model.invariant_point_attention import InvariantPointAttention


class StructureModuleLayer(nn.Module):
    def __init__(
        self,
        single_embedding_size,
        pair_embedding_size,
        dropout,
        ipa_embedding_size,
        num_query_points,
        num_value_points,
        num_heads,
    ):
        super().__init__()
        self.invariant_point_attention = InvariantPointAttention(
            single_embedding_size,
            pair_embedding_size,
            ipa_embedding_size,
            num_query_points,
            num_value_points,
            num_heads,
        )
        self.backbone_update = BackboneUpdate(single_embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(single_embedding_size)
        self.layer_norm2 = nn.LayerNorm(single_embedding_size)
        self.transition = nn.Sequential(
            nn.Linear(single_embedding_size, single_embedding_size),
            nn.ReLU(),
            nn.Linear(single_embedding_size, single_embedding_size),
            nn.ReLU(),
            nn.Linear(single_embedding_size, single_embedding_size),
        )

    @staticmethod
    def get_args(config):
        return {
            **InvariantPointAttention.get_args(config),
            "single_embedding_size": config.getint("General", "single_embedding_size"),
            "pair_embedding_size": config.getint("InputEmbedding", "pair_embedding_size"),
            "dropout": config.getfloat("StructureModule", "dropout"),
        }

    def forward(self, single, pair, frames, frames_truth=None, fape_clamp=None):
        single = single + self.invariant_point_attention(single, pair, frames)
        single = self.layer_norm1(self.dropout(single))
        single = single + self.transition(single)
        single = self.layer_norm2(self.dropout(single))
        frames = Frame.compose(frames, self.backbone_update(single))

        loss = (
            compute_fape_loss(
                frames,
                frames.translations,
                frames_truth,
                frames_truth.translations,
                eps=1e-12,
                clamp=fape_clamp,
            )
            if frames_truth is not None
            else None
        )

        return single, frames, loss


class StructureModule(nn.Module):
    def __init__(
        self,
        num_layers,
        single_embedding_size,
        pair_embedding_size,
        dropout,
        ipa_embedding_size,
        num_query_points,
        num_value_points,
        num_heads,
        num_lddt_bins,
        num_lddt_channels,
        device,
    ):
        super().__init__()
        self.structure_module_layer = StructureModuleLayer(
            single_embedding_size,
            pair_embedding_size,
            dropout,
            ipa_embedding_size,
            num_query_points,
            num_value_points,
            num_heads,
        )
        self.num_layers = num_layers
        self.single_layer_norm = nn.LayerNorm(single_embedding_size)
        self.pair_layer_norm = nn.LayerNorm(pair_embedding_size)
        self.single_linear = nn.Linear(single_embedding_size, single_embedding_size)
        self.lddt_predictor = LDDTPredictor(single_embedding_size, num_lddt_bins, num_lddt_channels)
        self.device = device

    def forward(self, single, pair, local_coords, frames_truth=None, fape_clamp=None):
        batch_dims = single.shape[:-1]
        single = self.single_layer_norm(single)
        pair = self.pair_layer_norm(pair)
        single = self.single_linear(single)
        frames = Frame(
            rotations=torch.eye(3, device=self.device).unsqueeze(0).repeat(*batch_dims, 1, 1),
            translations=torch.zeros(*batch_dims, 3, device=self.device),
        )

        aux_losses = []
        for i in range(self.num_layers):
            single, frames, loss = self.structure_module_layer(
                single, pair, frames, frames_truth, fape_clamp
            )
            aux_losses.append(loss)
            if i < self.num_layers - 1:
                frames.rotations = frames.rotations.detach()

        aux_loss = (
            torch.stack(aux_losses).mean() if all(l is not None for l in aux_losses) else None
        )
        fape_loss = (
            compute_fape_loss(
                frames,
                frames.translations,
                frames_truth,
                frames_truth.translations,
                clamp=fape_clamp,
            )
            if (frames_truth is not None)
            else None
        )
        batched_frames = Frame(frames.rotations.unsqueeze(-3), frames.translations.unsqueeze(-2))
        coords = Frame.apply(batched_frames, local_coords)

        chain_lddt, chain_plddt, fape_loss, conf_loss, aux_loss = None, None, None, None, None
        if frames_truth is not None:
            aux_loss = torch.stack(aux_losses).mean()
            fape_loss = compute_fape_loss(
                frames,
                frames.translations,
                frames_truth,
                frames_truth.translations,
                clamp=fape_clamp,
            )
            residue_LDDT_truth = self.lddt_predictor.compute_per_residue_LDDT(
                frames.translations, frames_truth.translations
            )
            conf_loss, chain_plddt = self.lddt_predictor(single, residue_LDDT_truth)
            chain_lddt = residue_LDDT_truth.mean(dim=-1)

        return coords, chain_plddt, chain_lddt, fape_loss, conf_loss, aux_loss

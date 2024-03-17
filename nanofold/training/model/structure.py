import torch
from torch import nn

from nanofold.training.loss import compute_fape_loss
from nanofold.training.frame import Frame
from nanofold.training.model.backbone_update import BackboneUpdate
from nanofold.training.model.invariant_point_attention import InvariantPointAttention


class StructureModuleLayer(nn.Module):
    def __init__(
        self,
        single_embedding_size,
        pair_embedding_size,
        dropout,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.invariant_point_attention = InvariantPointAttention(
            single_embedding_size=single_embedding_size,
            pair_embedding_size=pair_embedding_size,
            *args,
            **kwargs,
        )
        self.backbone_update = BackboneUpdate(single_embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(single_embedding_size)
        self.layer_norm2 = nn.LayerNorm(single_embedding_size)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(single_embedding_size, single_embedding_size)
        self.linear2 = nn.Linear(single_embedding_size, single_embedding_size)
        self.linear3 = nn.Linear(single_embedding_size, single_embedding_size)

    @staticmethod
    def get_args(config):
        return {
            **InvariantPointAttention.get_args(config),
            "single_embedding_size": config.getint("General", "single_embedding_size"),
            "pair_embedding_size": config.getint("InputEmbedding", "pair_embedding_size"),
            "dropout": config.getfloat("StructureModule", "dropout"),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**cls.get_args(config))

    def forward(self, single, pair, frames, frames_truth=None):
        single = single + self.invariant_point_attention(single, pair, frames)
        single = self.layer_norm1(self.dropout(single))
        single = single + self.linear3(self.relu(self.linear2(self.relu(self.linear1(single)))))
        single = self.layer_norm2(self.dropout(single))
        frames = Frame.compose(frames, self.backbone_update(single))

        loss = (
            compute_fape_loss(
                frames, frames.translations, frames_truth, frames_truth.translations, eps=1e-12
            )
            if frames_truth is not None
            else None
        )

        return single, frames, loss


class StructureModule(nn.Module):
    def __init__(self, num_layers, single_embedding_size, pair_embedding_size, *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                StructureModuleLayer(
                    single_embedding_size=single_embedding_size,
                    pair_embedding_size=pair_embedding_size,
                    *args,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_layer_norm = nn.LayerNorm(single_embedding_size)
        self.pair_layer_norm = nn.LayerNorm(pair_embedding_size)
        self.single_linear = nn.Linear(single_embedding_size, single_embedding_size)

    @staticmethod
    def get_args(config):
        return {
            **StructureModuleLayer.get_args(config),
            "num_layers": config.getint("StructureModule", "num_layers"),
            "single_embedding_size": config.getint("General", "single_embedding_size"),
            "pair_embedding_size": config.getint("InputEmbedding", "pair_embedding_size"),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**cls.get_args(config))

    def forward(self, single, pair, local_coords, frames_truth=None):
        batch_dims = single.shape[:-1]
        single = self.single_layer_norm(single)
        pair = self.pair_layer_norm(pair)
        single = self.single_linear(single)
        frames = Frame(
            rotations=torch.eye(3).unsqueeze(0).repeat(*batch_dims, 1, 1),
            translations=torch.zeros(*batch_dims, 3),
        )

        aux_losses = None
        for i, layer in enumerate(self.layers):
            single, frames, loss = layer(single, pair, frames, frames_truth)
            if loss is not None:
                aux_losses = torch.empty(0) if aux_losses is None else aux_losses
                aux_losses = torch.cat([aux_losses, loss.unsqueeze(-1)], dim=-1)
            if i < len(self.layers) - 1:
                frames.rotations = frames.rotations.detach()

        aux_loss = aux_losses.mean(dim=-1) if aux_losses is not None else None
        fape_loss = (
            compute_fape_loss(frames, frames.translations, frames_truth, frames_truth.translations)
            if (frames_truth is not None)
            else None
        )
        batched_frames = Frame(frames.rotations.unsqueeze(-3), frames.translations.unsqueeze(-2))
        coords = Frame.apply(batched_frames, local_coords)

        return coords, fape_loss, aux_loss

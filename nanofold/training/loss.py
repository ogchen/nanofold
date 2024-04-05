import torch
from torch import nn

from nanofold.training.frame import Frame


def compute_fape_loss(
    frames, coords, frames_truth, coords_truth, eps=1e-4, length_scale=10.0, clamp=10.0
):
    """Compute the frame-aligned point error (FAPE) between two sets of frames and coordinates.
    Args:
        frames (Frame): The predicted frames.
        coords (torch.Tensor): The predicted coordinates.
        frames_truth (Frame): The ground truth frames.
        coords_truth (torch.Tensor): The ground truth coordinates.
        eps (float): A small value to ensure gradients are well behaved for small differences.
        length_scale (float): A scaling factor for the loss.
        clamp (float): The maximum value for a pairwise loss.
    """
    inverse = Frame.inverse(frames)
    inverse.rotations = inverse.rotations.unsqueeze(-3)
    inverse.translations = inverse.translations.unsqueeze(-2)
    inverse_truth = Frame.inverse(frames_truth)
    inverse_truth.rotations = inverse_truth.rotations.unsqueeze(-3)
    inverse_truth.translations = inverse_truth.translations.unsqueeze(-2)
    globals = Frame.apply(inverse, coords.unsqueeze(-3))
    globals_truth = Frame.apply(inverse_truth, coords_truth.unsqueeze(-3))
    difference = globals - globals_truth
    squared_norm = difference.unsqueeze(-2) @ difference.unsqueeze(-1)
    norm = torch.sqrt(squared_norm + eps).squeeze(-1).squeeze(-1)
    if clamp is not None:
        norm = torch.clamp(norm, max=clamp)
    return (1 / length_scale) * norm.mean()


class DistogramLoss(nn.Module):
    def __init__(self, pair_embedding_size, num_channels=12):
        super().__init__()
        self.bins = torch.arange(2, 22, 20 / 64)
        self.projection = nn.Sequential(
            nn.LayerNorm(pair_embedding_size),
            nn.Linear(pair_embedding_size, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, len(self.bins)),
        )

    def forward(self, pair_rep, coords_truth):
        logits = self.projection(pair_rep)
        distance_mat = torch.norm(coords_truth.unsqueeze(-2) - coords_truth.unsqueeze(-3), dim=-1)
        index = torch.argmin(torch.abs(distance_mat.unsqueeze(-1) - self.bins), dim=-1)
        loss = nn.functional.cross_entropy(logits.transpose(-1, 1), index)
        return loss


class LDDTPredictor(nn.Module):
    def __init__(self, single_embedding_size, num_channels=12):
        super().__init__()
        self.bins = torch.arange(1, 100, 2).float()
        self.activate = nn.Sequential(
            nn.LayerNorm(single_embedding_size),
            nn.Linear(single_embedding_size, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, len(self.bins)),
        )

    def compute_per_residue_LDDT(self, coords, coords_truth, cutoff=15.0, eps=1e-10):
        distance_mat = torch.norm(coords.unsqueeze(-2) - coords.unsqueeze(-3), dim=-1)
        distance_mat_truth = torch.norm(
            coords_truth.unsqueeze(-2) - coords_truth.unsqueeze(-3), dim=-1
        )
        difference = torch.abs(distance_mat - distance_mat_truth)
        score_mask = distance_mat_truth < cutoff
        torch.diagonal(score_mask, dim1=-1).zero_()
        score_mat = 0.25 * (
            (difference < 0.5).float()
            + (difference < 1.0).float()
            + (difference < 2.0).float()
            + (difference < 4.0).float()
        )
        scale_factor = 1 / (torch.sum(score_mask, dim=-1) + eps)
        score = scale_factor * (torch.sum(score_mask * score_mat, dim=-1) + eps)
        return score

    def forward(self, single, residue_LDDT_truth=None):
        logits = self.activate(single)
        conf_loss, chain_plddt = None, None

        if not self.training:
            chain_plddt = torch.mean(nn.functional.softmax(logits, dim=-1) @ self.bins, dim=-1)

        if residue_LDDT_truth is not None:
            index = (
                torch.floor(len(self.bins) * residue_LDDT_truth)
                .long()
                .clamp(max=len(self.bins) - 1)
            )
            conf_loss = nn.functional.cross_entropy(logits.transpose(-1, 1), index)

        return conf_loss, chain_plddt

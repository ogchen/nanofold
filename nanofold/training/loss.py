import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_smooth_lddt_loss(x, x_gt):
    dist = torch.linalg.vector_norm(x.unsqueeze(-3) - x.unsqueeze(-2), dim=-1)
    dist_gt = torch.linalg.vector_norm(x_gt.unsqueeze(-3) - x_gt.unsqueeze(-2), dim=-1)
    diff = torch.abs(dist - dist_gt)
    e = 0.25 * (
        F.sigmoid(0.5 - diff) + F.sigmoid(1 - diff) + F.sigmoid(2 - diff) + F.sigmoid(4 - diff)
    )
    mask = dist_gt < 15.0
    torch.diagonal(mask, dim1=-2, dim2=-1).zero_()
    return torch.sum(mask * e, dim=(-2, -1), keepdim=True) / torch.sum(
        mask, dim=(-2, -1), keepdim=True
    )


class DistogramLoss(nn.Module):
    def __init__(self, pair_embedding_size, num_bins, device):
        super().__init__()
        self.bins = torch.arange(2, 22, 20 / num_bins, device=device)
        self.projection = nn.Linear(pair_embedding_size, len(self.bins))

    def forward(self, pair_rep, coords_truth):
        logits = self.projection(pair_rep + pair_rep.transpose(-3, -2))
        distance_mat = torch.norm(coords_truth.unsqueeze(-2) - coords_truth.unsqueeze(-3), dim=-1)
        index = torch.argmin(torch.abs(distance_mat.unsqueeze(-1) - self.bins), dim=-1)
        loss = nn.functional.cross_entropy(logits.transpose(-1, 1), index)
        return loss

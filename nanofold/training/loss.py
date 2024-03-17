import torch
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
    globals = Frame.apply(inverse, coords)
    globals_truth = Frame.apply(inverse_truth, coords_truth)
    difference = globals - globals_truth
    squared_norm = difference.unsqueeze(-2) @ difference.unsqueeze(-1)
    norm = torch.sqrt(squared_norm + eps).clamp(max=clamp).squeeze(-1).squeeze(-1)
    return (1 / length_scale) * norm.mean(dim=(-2, -1))

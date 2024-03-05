import torch
from nanofold.frame import Frame


def loss_fape(
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
    inverse_truth = Frame.inverse(frames_truth)
    globals = Frame.apply(inverse, coords)
    globals_truth = Frame.apply(inverse_truth, coords_truth)
    squared_norm = torch.linalg.vector_norm(globals - globals_truth, dim=-1) ** 2
    norm = torch.sqrt(squared_norm + eps).clamp(max=clamp)
    return (1 / length_scale) * norm.mean()

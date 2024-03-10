import math
import torch
from nanofold.frame import Frame
from nanofold.loss import compute_fape_loss


def test_loss_fape():
    cos = math.cos(1)
    sin = math.sin(1)
    frames = Frame(
        rotations=torch.stack(
            [
                torch.eye(3),
                torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]),
            ]
        ),
        translations=torch.stack([torch.zeros(3), torch.tensor([20, 10, 10])]),
    )
    coords = torch.stack([torch.zeros(3), torch.tensor([10, 20, 30]), torch.tensor([1, 2, 3])])
    frames_truth = Frame(
        torch.stack([torch.eye(3), torch.eye(3)]),
        torch.stack([torch.zeros(3), torch.zeros(3)]),
    )
    coords_truth = 10 * torch.ones(len(coords), 3)
    length_scale = 10
    clamp = 20
    loss = compute_fape_loss(
        frames, coords, frames_truth, coords_truth, length_scale=length_scale, clamp=clamp
    )

    expected = 0
    for i in range(len(frames)):
        for j in range(len(coords)):
            inverse = Frame.inverse(frames[i])
            inverse_truth = Frame.inverse(frames_truth[i])
            glob_coord = Frame.apply(inverse, coords[j])
            glob_coord_truth = Frame.apply(inverse_truth, coords_truth[j])
            expected += min(clamp, abs(glob_coord - glob_coord_truth).norm())
    expected /= len(frames) * len(coords) * length_scale
    assert torch.isclose(loss, expected)

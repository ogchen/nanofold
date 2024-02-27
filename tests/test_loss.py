import torch
from nanofold.frame import Frame
from nanofold.loss import loss_fape


def test_loss_fape():
    frames = Frame(
        rotations=torch.stack(
            [
                torch.eye(3),
                torch.tensor([[-0.5, -1.0, 0.0], [1, -0.5, 0.0], [0.0, 0.0, 1.0]]),
            ]
        ),
        translations=torch.stack([torch.zeros(3), torch.tensor([20, 10, 10])]),
    )
    coords = torch.stack([torch.zeros(3), torch.tensor([10, 20, 30])])
    frames_truth = Frame(
        torch.stack([torch.eye(3), torch.eye(3)]),
        torch.stack([torch.zeros(3), torch.zeros(3)]),
    )
    coords_truth = 10 * torch.ones(2, 3)

    loss = loss_fape(frames, coords, frames_truth, coords_truth, clamp=20)

    expected = 0.1 * torch.sqrt(torch.tensor([300, 20**2, 20**2, 140])).mean()
    assert torch.isclose(loss, expected)

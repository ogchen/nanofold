from nanofold.train.loss import compute_diffusion_loss

import torch


def test_diffusion_loss():
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    x_truth = torch.tensor([[10.0, 11.0, 10.0], [9.0, 10.0, 10.0], [10.0, 10.0, 11.0]])
    loss = compute_diffusion_loss(x, x_truth, t=1.0, data_std_dev=16)
    assert torch.isclose(loss["mse_loss"], torch.zeros(1))
    assert torch.isclose(loss["lddt_loss"], torch.zeros(1))
    assert torch.isclose(loss["diffusion_loss"], torch.zeros(1))

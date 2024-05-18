from nanofold.train.util import rigid_align

import torch


def test_rigid_align():
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    x_truth = torch.tensor([[10.0, 11.0, 10.0], [9.0, 10.0, 10.0], [10.0, 10.0, 11.0]])
    x_aligned = rigid_align(x, x_truth)
    assert torch.allclose(x_aligned, x_truth, atol=1e-4)

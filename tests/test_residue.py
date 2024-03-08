from nanofold import residue
import torch


def test_compute_residue_rotation():
    n = torch.tensor([101.601, 38.534, -1.962])
    ca = torch.tensor([103.062, 38.513, -2.159])
    c = torch.tensor([103.354, 38.323, -3.656])
    result = residue.compute_residue_rotation(n, ca, c)
    assert torch.allclose(result @ result.transpose(-2, -1), torch.eye(3), atol=1e-5)

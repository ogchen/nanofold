from nanofold import residue
import torch


def test_compute_residue_frames():
    num_residues = 5
    coords = torch.rand(num_residues, 3, 3)
    frames = residue.compute_residue_frames(coords)
    frames.rotations
    assert torch.allclose(
        frames.rotations @ frames.rotations.transpose(-2, -1), torch.eye(3), atol=1e-5
    )
    assert torch.allclose(frames.translations, coords[:, 1, :])

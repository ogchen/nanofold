from nanofold.training import residue
import torch


def test_compute_backbone_coords():
    sequence = ["MET", "PHE", "PRO", "SER", "THR"]
    sequence_1l = "MFPST"
    frames = residue.Frame(
        rotations=torch.eye(3).unsqueeze(0).repeat(len(sequence), 1, 1),
        translations=torch.arange(len(sequence) * 3).view(len(sequence), 3).float(),
    )
    coords = residue.compute_backbone_coords(frames, sequence)
    assert coords.shape == (len(sequence), 3, 3)
    assert torch.allclose(coords[:, 1, :], frames.translations)
    assert torch.allclose(coords, residue.compute_backbone_coords(frames, sequence_1l))

import numpy as np
from nanofold.preprocess import residue


def test_compute_residue_frames():
    num_residues = 5
    coords = np.random.rand(num_residues, 3, 3)
    rotations, translations = residue.compute_residue_frames(coords)
    assert np.allclose(rotations @ np.swapaxes(rotations, -2, -1), np.eye(3), atol=1e-5)
    assert np.allclose(translations, coords[:, 1, :])

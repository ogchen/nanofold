import numpy as np


def compute_residue_frames(coords):
    x0 = coords[:, 0, :]
    x1 = coords[:, 1, :]
    x2 = coords[:, 2, :]
    v0 = x2 - x1
    v1 = x0 - x1
    e0 = v0 / np.linalg.norm(v0, axis=-1, keepdims=True)
    dot = e0[..., np.newaxis, :] @ v1[..., np.newaxis]
    u1 = v1 - e0 * np.squeeze(dot, axis=-1)

    e1 = u1 / np.linalg.norm(u1, axis=-1, keepdims=True)
    e2 = np.cross(e0, e1)
    rotations = np.stack([e0, e1, e2], axis=-2)
    return rotations, x1

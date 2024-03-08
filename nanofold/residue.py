import torch
from nanofold.frame import Frame

RESIDUE_LIST = [
    ("A", "ALA"),
    ("R", "ARG"),
    ("N", "ASN"),
    ("D", "ASP"),
    ("C", "CYS"),
    ("Q", "GLN"),
    ("E", "GLU"),
    ("G", "GLY"),
    ("H", "HIS"),
    ("I", "ILE"),
    ("L", "LEU"),
    ("K", "LYS"),
    ("M", "MET"),
    ("F", "PHE"),
    ("P", "PRO"),
    ("S", "SER"),
    ("T", "THR"),
    ("W", "TRP"),
    ("Y", "TYR"),
    ("V", "VAL"),
]


def compute_residue_frames(coords):
    x0 = coords[:, 0, :]
    x1 = coords[:, 1, :]
    x2 = coords[:, 2, :]
    v0 = x2 - x1
    v1 = x0 - x1
    e0 = v0 / torch.linalg.vector_norm(v0, dim=-1, keepdim=True)
    dot = e0.unsqueeze(-2) @ v1.unsqueeze(-1)
    u1 = v1 - e0 * dot.squeeze(-1)

    e1 = u1 / torch.linalg.vector_norm(u1, dim=-1, keepdim=True)
    e2 = torch.linalg.cross(e0, e1)
    rotations = torch.stack([e0, e1, e2], dim=-2)
    return Frame(rotations=rotations, translations=x1)

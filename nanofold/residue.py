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

BACKBONE_POSITIONS = {
    "ALA": [
        ("N", torch.tensor([-0.525, 1.363, 0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.526, -0.000, -0.000])),
    ],
    "ARG": [
        ("N", torch.tensor([-0.524, 1.362, -0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.525, -0.000, -0.000])),
    ],
    "ASN": [
        ("N", torch.tensor([-0.536, 1.357, 0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.526, -0.000, -0.000])),
    ],
    "ASP": [
        ("N", torch.tensor([-0.525, 1.362, -0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.527, 0.000, -0.000])),
    ],
    "CYS": [
        ("N", torch.tensor([-0.522, 1.362, -0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.524, 0.000, 0.000])),
    ],
    "GLN": [
        ("N", torch.tensor([-0.526, 1.361, -0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.526, 0.000, 0.000])),
    ],
    "GLU": [
        ("N", torch.tensor([-0.528, 1.361, 0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.526, -0.000, -0.000])),
    ],
    "GLY": [
        ("N", torch.tensor([-0.572, 1.337, 0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.517, -0.000, -0.000])),
    ],
    "HIS": [
        ("N", torch.tensor([-0.527, 1.360, 0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.525, 0.000, 0.000])),
    ],
    "ILE": [
        ("N", torch.tensor([-0.493, 1.373, -0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.527, -0.000, -0.000])),
    ],
    "LEU": [
        ("N", torch.tensor([-0.520, 1.363, 0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.525, -0.000, -0.000])),
    ],
    "LYS": [
        ("N", torch.tensor([-0.526, 1.362, -0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.526, 0.000, 0.000])),
    ],
    "MET": [
        ("N", torch.tensor([-0.521, 1.364, -0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.525, 0.000, 0.000])),
    ],
    "PHE": [
        ("N", torch.tensor([-0.518, 1.363, 0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.524, 0.000, -0.000])),
    ],
    "PRO": [
        ("N", torch.tensor([-0.566, 1.351, -0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.527, -0.000, 0.000])),
    ],
    "SER": [
        ("N", torch.tensor([-0.529, 1.360, -0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.525, -0.000, -0.000])),
    ],
    "THR": [
        ("N", torch.tensor([-0.517, 1.364, 0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.526, 0.000, -0.000])),
    ],
    "TRP": [
        ("N", torch.tensor([-0.521, 1.363, 0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.525, -0.000, 0.000])),
    ],
    "TYR": [
        ("N", torch.tensor([-0.522, 1.362, 0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.524, -0.000, -0.000])),
    ],
    "VAL": [
        ("N", torch.tensor([-0.494, 1.373, -0.000])),
        ("CA", torch.tensor([0.000, 0.000, 0.000])),
        ("C", torch.tensor([1.527, -0.000, -0.000])),
    ],
}


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


def compute_backbone_coords(frames, sequence):
    if isinstance(sequence, str):
        residue_lookup = dict(RESIDUE_LIST)
        sequence = [residue_lookup[r] for r in sequence]
    if len(sequence) != len(frames):
        raise ValueError("Sequence length must match number of frames")

    local_coords = torch.stack(
        [torch.stack([a[1] for a in BACKBONE_POSITIONS[r]]) for r in sequence]
    )
    return Frame.apply(frames.unsqueeze(1), local_coords)

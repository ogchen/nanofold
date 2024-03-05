import torch

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


def compute_residue_rotation(n_coords, ca_coords, c_coords):
    if n_coords is None or c_coords is None:
        return torch.eye(3)
    ca_c = c_coords - ca_coords
    ca_n = n_coords - ca_coords
    ca_c_unit = ca_c / ca_c.norm()
    ca_n_unit = ca_n / ca_n.norm()
    cross = torch.linalg.cross(ca_c_unit, ca_n_unit)
    cross_unit = cross / cross.norm()

    reference_mat = torch.tensor(
        [[1, 0, 0], [ca_c_unit.dot(ca_n_unit), cross.norm(), 0], [0, 0, 1]]
    ).transpose(-2, -1)
    mat = torch.stack([ca_c_unit, ca_n_unit, cross_unit]).transpose(-2, -1)
    return mat @ reference_mat.inverse()

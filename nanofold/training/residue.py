import torch

from nanofold.common.residue_definitions import BACKBONE_POSITIONS
from nanofold.common.residue_definitions import RESIDUE_LOOKUP_1L
from nanofold.training.frame import Frame


def compute_backbone_coords(frames, sequence):
    if isinstance(sequence, str):
        sequence = [RESIDUE_LOOKUP_1L[r] for r in sequence]
    if len(sequence) != len(frames):
        raise ValueError("Sequence length must match number of frames")

    local_coords = torch.tensor([[a[1] for a in BACKBONE_POSITIONS[r]] for r in sequence])
    return Frame.apply(frames.unsqueeze(1), local_coords)

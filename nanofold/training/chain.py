from nanofold.training.frame import Frame
from nanofold.training.residue import RESIDUE_LIST


class Chain:
    RESIDUE_LOOKUP = {r[1]: r[0] for r in RESIDUE_LIST}

    def __init__(self, id, chain, frames, sequence, positions):
        self.id = id
        self.chain = chain
        self.frames = frames
        self.sequence = sequence
        self.positions = positions

    @classmethod
    def from_residue_list(cls, id, residue_list, frames):
        chain = cls(id, chain=[], frames=frames, sequence="", positions=[])
        chain = [{"resname": r["resname"], "id": r["id"]} for r in residue_list]
        sequence = "".join([cls.RESIDUE_LOOKUP[r["resname"]] for r in residue_list])
        positions = [r["id"][-1][1] for r in residue_list]
        return cls(id, chain, frames, sequence, positions)

    def __repr__(self):
        return f"Chain(id={self.id}, sequence={self.sequence})"

    def __len__(self):
        return len(self.chain)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Chain(
                self.id,
                self.chain[key],
                self.frames[key],
                self.sequence[key],
                self.positions[key],
            )

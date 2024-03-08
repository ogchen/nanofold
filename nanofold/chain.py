from nanofold.frame import Frame
from nanofold.residue import RESIDUE_LIST


class Chain:
    RESIDUE_LOOKUP = {r[1]: r[0] for r in RESIDUE_LIST}

    def __init__(self, id, chain, frames, sequence, positions):
        self.id = id
        self.chain = chain
        self.frames = frames
        self.sequence = sequence
        self.positions = positions

    @classmethod
    def from_residue_list(cls, id, residue_list):
        chain = cls(id, chain=[], frames=Frame(), sequence="", positions=[])
        for residue in residue_list:
            chain.add_residue(residue)
        return chain

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

    def add_residue(self, residue):
        self.chain.append(
            {
                "resname": residue["resname"],
                "id": residue["id"],
            }
        )
        self.frames += Frame(residue["rotation"], residue["translation"])
        self.sequence += Chain.RESIDUE_LOOKUP[residue["resname"]]
        self.positions.append(residue["id"][-1][1])

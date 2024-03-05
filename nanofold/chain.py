from nanofold.frame import Frame
from nanofold.residue import RESIDUE_LIST


class Chain:
    def __init__(self, id, residue_list):
        self.id = id
        self.chain = []
        self.frames = Frame()

        for residue in residue_list:
            self.add_residue(residue)

        residue_lookup = {r[1]: r[0] for r in RESIDUE_LIST}
        self.sequence = "".join([residue_lookup[r["resname"]] for r in self.chain])

    def __repr__(self):
        return f"Chain(id={self.id}, sequence={self.sequence})"

    def __len__(self):
        return len(self.chain)

    def add_residue(self, residue):
        self.chain.append(
            {
                "resname": residue["resname"],
                "id": residue["id"],
            }
        )
        self.frames += Frame(residue["rotation"], residue["translation"])

from nanofold.frame import Frame


class Chain:
    def __init__(self, id, sequence, residue_list):
        self.id = id
        self.sequence = sequence
        self.chain = []
        self.frames = Frame()

        for residue in residue_list:
            self.add_residue(residue)

        if len(self.sequence) != len(self.chain):
            raise RuntimeError(f"Sequence length mismatch for chain {self.id}")

    def add_residue(self, residue):
        self.chain.append({"resname": residue["resname"], "id": residue["id"]})
        self.frames += Frame(residue["rotation"], residue["translation"])

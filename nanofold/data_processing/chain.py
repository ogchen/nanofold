from nanofold.common.residue_definitions import RESIDUE_LOOKUP_3L


class Chain:
    def __init__(self, id, chain, rotations, translations, sequence, positions):
        self.id = id
        self.chain = chain
        self.rotations = rotations
        self.translations = translations
        self.sequence = sequence
        self.positions = positions

    @classmethod
    def from_residue_list(cls, id, residue_list, rotations, translations):
        chain = cls(
            id, chain=[], rotations=rotations, translations=translations, sequence="", positions=[]
        )
        chain = [{"resname": r["resname"], "id": r["id"]} for r in residue_list]
        sequence = "".join([RESIDUE_LOOKUP_3L[r["resname"]] for r in residue_list])
        positions = [r["id"][-1][1] for r in residue_list]
        return cls(id, chain, rotations, translations, sequence, positions)

    def __repr__(self):
        return f"Chain(id={self.id}, sequence={self.sequence})"

    def __len__(self):
        return len(self.chain)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Chain(
                self.id,
                self.chain[key],
                self.rotations[key],
                self.translations[key],
                self.sequence[key],
                self.positions[key],
            )

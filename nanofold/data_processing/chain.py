from nanofold.common.residue_definitions import RESIDUE_LOOKUP_3L


class Chain:
    def __init__(self, id, release_date, chain, rotations, translations, sequence, positions):
        self.id = id
        self.release_date = release_date
        self.chain = chain
        self.rotations = rotations
        self.translations = translations
        self.sequence = sequence
        self.positions = positions

    @classmethod
    def from_residue_list(cls, id, release_date, residue_list, rotations, translations):
        chain = [{"resname": r["resname"], "id": r["id"]} for r in residue_list]
        sequence = "".join([RESIDUE_LOOKUP_3L[r["resname"]] for r in residue_list])
        positions = [r["id"][-1][1] for r in residue_list]
        return cls(id, release_date, chain, rotations, translations, sequence, positions)

    def __repr__(self):
        return f"Chain(id={self.id}, release={self.release_date}, sequence={self.sequence})"

    def __len__(self):
        return len(self.chain)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Chain(
                self.id,
                self.release_date,
                self.chain[key],
                self.rotations[key],
                self.translations[key],
                self.sequence[key],
                self.positions[key],
            )

    @staticmethod
    def to_record(chain):
        return {
            "model_id": chain.id[0],
            "chain_id": chain.id[1],
            "release_date": chain.release_date,
            "chain": chain.chain,
            "rotations": chain.rotations.flatten().tolist(),
            "translations": chain.translations.flatten().tolist(),
            "sequence": chain.sequence,
            "positions": chain.positions,
        }

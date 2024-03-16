import pyarrow as pa

from nanofold.common.residue_definitions import RESIDUE_LOOKUP_3L


class Chain:
    SCHEMA = pa.schema(
        [
            ("model_id", pa.string()),
            ("chain_id", pa.string()),
            ("release_date", pa.string()),
            ("rotations", pa.list_(pa.float32())),
            ("translations", pa.list_(pa.float32())),
            ("sequence", pa.string()),
            ("positions", pa.list_(pa.int32())),
        ]
    )

    def __init__(self, id, release_date, rotations, translations, sequence, positions):
        self.id = id
        self.release_date = release_date
        self.rotations = rotations
        self.translations = translations
        self.sequence = sequence
        self.positions = positions

    @classmethod
    def from_residue_list(cls, id, release_date, residue_list, rotations, translations):
        sequence = "".join([RESIDUE_LOOKUP_3L[r["resname"]] for r in residue_list])
        positions = [r["id"][-1][1] for r in residue_list]
        return cls(id, release_date, rotations, translations, sequence, positions)

    def __repr__(self):
        return f"Chain(id={self.id}, release={self.release_date}, sequence={self.sequence})"

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Chain(
                self.id,
                self.release_date,
                self.rotations[key],
                self.translations[key],
                self.sequence[key],
                self.positions[key],
            )

    @staticmethod
    def to_record_batch(chains):
        data = [
            pa.array([c.id[0] for c in chains]),
            pa.array([c.id[1] for c in chains]),
            pa.array([c.release_date for c in chains]),
            pa.array([c.rotations.flatten() for c in chains]),
            pa.array([c.translations.flatten() for c in chains]),
            pa.array([c.sequence for c in chains]),
            pa.array([c.positions for c in chains]),
        ]
        return pa.record_batch(data, schema=Chain.SCHEMA)

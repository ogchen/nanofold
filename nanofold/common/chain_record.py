import pyarrow as pa


class ChainRecord:
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

    def __init__(
        self, model_id, chain_id, release_date, rotations, translations, sequence, positions
    ):
        self.model_id = model_id
        self.chain_id = chain_id
        self.release_date = release_date
        self.rotations = rotations
        self.translations = translations
        self.sequence = sequence
        self.positions = positions

    def __repr__(self):
        return f"ChainRecord(model_id={self.model_id}, chain_id={self.chain_id}, release={self.release_date}, sequence={self.sequence})"

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, key):
        return ChainRecord(
            self.model_id,
            self.chain_id,
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
        return pa.record_batch(data, schema=ChainRecord.SCHEMA)

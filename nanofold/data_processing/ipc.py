import logging
import pyarrow as pa
from itertools import batched

SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.string()),
        pa.field("chain_id", pa.string()),
        pa.field("rotations", pa.list_(pa.list_(pa.list_(pa.float32())))),
        pa.field("translations", pa.list_(pa.list_(pa.float32()))),
        pa.field("sequence", pa.string()),
        pa.field("positions", pa.list_(pa.int32())),
        pa.field(
            "msa",
            pa.struct(
                [
                    pa.field("alignments", pa.list_(pa.string())),
                    pa.field("deletion_matrix", pa.list_(pa.list_(pa.int32()))),
                ]
            ),
        ),
    ]
)


def get_ready_chains(db_manager):
    return db_manager.chains().find({"msa": {"$exists": 1}})


def dump_to_ipc(db_manager, output, batch_size=2000):
    chains = get_ready_chains(db_manager)

    num_chains = 0
    with pa.OSFile(str(output), mode="w") as f:
        with pa.ipc.new_file(f, SCHEMA) as writer:
            for batch in batched(chains, batch_size):
                record_batch = {
                    "structure_id": pa.array([c["_id"]["structure_id"] for c in batch]),
                    "chain_id": pa.array([c["_id"]["chain_id"] for c in batch]),
                    "rotations": pa.array([c["rotations"] for c in batch]),
                    "translations": pa.array([c["translations"] for c in batch]),
                    "sequence": pa.array([c["sequence"] for c in batch]),
                    "positions": pa.array([c["positions"] for c in batch]),
                    "msa": pa.array([c["msa"] for c in batch]),
                }
                record_batch = pa.RecordBatch.from_arrays(
                    list(record_batch.values()), schema=SCHEMA
                )
                writer.write_batch(record_batch)
                num_chains += len(batch)
                logging.info(f"Wrote {num_chains} chains to IPC file")

    logging.info(f"Finished writing {num_chains} chains to {output}")

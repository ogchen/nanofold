import logging
import pickle
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
        pa.field("cluster_msa", pa.list_(pa.list_(pa.list_(pa.float32())))),
        pa.field("msa_feat", pa.list_(pa.list_(pa.list_(pa.float32())))),
        pa.field("extra_msa_feat", pa.list_(pa.list_(pa.list_(pa.float32())))),
    ]
)


def get_ready_chains(db_manager):
    return db_manager.chains().find({"msa_feat": {"$exists": 1}})


def dump_to_ipc(db_manager, msa_output_dir, output, batch_size=50):
    chains = get_ready_chains(db_manager)

    num_chains = 0
    with pa.OSFile(str(output), mode="w") as f:
        with pa.ipc.new_file(f, SCHEMA) as writer:
            for batch in batched(chains, batch_size):
                msa_features = []
                for c in batch:
                    with open(
                        msa_output_dir / f"{c['_id']['structure_id']}_{c['_id']['chain_id']}.pkl",
                        "rb",
                    ) as f:
                        msa_features.append(pickle.load(f))

                record_batch = {
                    "structure_id": pa.array([c["_id"]["structure_id"] for c in batch]),
                    "chain_id": pa.array([c["_id"]["chain_id"] for c in batch]),
                    "rotations": pa.array([c["rotations"] for c in batch]),
                    "translations": pa.array([c["translations"] for c in batch]),
                    "sequence": pa.array([c["sequence"] for c in batch]),
                    "positions": pa.array([c["positions"] for c in batch]),
                    "cluster_msa": pa.array([m["cluster_msa"].tolist() for m in msa_features]),
                    "msa_feat": pa.array([m["msa_feat"].tolist() for m in msa_features]),
                    "extra_msa_feat": pa.array(
                        [m["extra_msa_feat"].tolist() for m in msa_features]
                    ),
                }
                record_batch = pa.RecordBatch.from_arrays(
                    list(record_batch.values()), schema=SCHEMA
                )
                writer.write_batch(record_batch)
                num_chains += len(batch)
                logging.info(f"Wrote {num_chains} chains to IPC file")

    logging.info(f"Finished writing {num_chains} chains to {output}")

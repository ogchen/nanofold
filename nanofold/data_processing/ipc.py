import glob
import gzip
import logging
import os
import pickle
import pyarrow as pa
from functools import partial
from itertools import batched
from pathlib import Path

SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.string()),
        pa.field("chain_id", pa.string()),
        pa.field("rotations", pa.list_(pa.list_(pa.list_(pa.float32())))),
        pa.field("translations", pa.list_(pa.list_(pa.float32()))),
        pa.field("sequence", pa.string()),
        pa.field("positions", pa.list_(pa.int16())),
        pa.field("template_mask", pa.list_(pa.list_(pa.bool_()))),
        pa.field("template_sequence", pa.list_(pa.string())),
        pa.field("template_translations", pa.list_(pa.list_(pa.list_(pa.float32())))),
        pa.field("cluster_msa", pa.list_(pa.list_(pa.list_(pa.bool_())))),
        pa.field("cluster_has_deletion", pa.list_(pa.list_(pa.bool_()))),
        pa.field("cluster_deletion_value", pa.list_(pa.list_(pa.float32()))),
        pa.field("cluster_deletion_mean", pa.list_(pa.list_(pa.float32()))),
        pa.field("cluster_profile", pa.list_(pa.list_(pa.list_(pa.float32())))),
        pa.field("extra_msa", pa.list_(pa.list_(pa.list_(pa.bool_())))),
        pa.field("extra_msa_has_deletion", pa.list_(pa.list_(pa.bool_()))),
        pa.field("extra_msa_deletion_value", pa.list_(pa.list_(pa.float32()))),
    ]
)


def get_ready_chains(db_manager, msa_output_dir):
    chains = db_manager.chains().find({"templates": {"$exists": True}})
    search_glob = os.path.join(msa_output_dir, "*.pkl.gz")
    msa_files = glob.glob(search_glob)
    found_ids = [Path(m).stem.split(".")[0] for m in msa_files]
    for c in chains:
        if f"{c['_id']['structure_id']}_{c['_id']['chain_id']}" in found_ids:
            yield c


def get_msa_features(msa_output_dir, chain):
    try:
        with gzip.open(
            msa_output_dir / f"{chain['_id']['structure_id']}_{chain['_id']['chain_id']}.pkl.gz",
            "rb",
        ) as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading MSA features for {chain['_id']}: {e}")
        return None


def get_record_batch(executor, msa_feat_getter, chain_batch):
    batch = [
        (c, m)
        for c, m in zip(chain_batch, list(executor.map(msa_feat_getter, chain_batch)))
        if m is not None
    ]

    return [
        pa.array([c["_id"]["structure_id"] for c, _ in batch]),
        pa.array([c["_id"]["chain_id"] for c, _ in batch]),
        pa.array([c["rotations"] for c, _ in batch]),
        pa.array([c["translations"] for c, _ in batch]),
        pa.array([c["sequence"] for c, _ in batch]),
        pa.array([c["positions"] for c, _ in batch]),
        pa.array([c["templates"]["mask"] for c, _ in batch]),
        pa.array([c["templates"]["sequence"] for c, _ in batch]),
        pa.array([c["templates"]["translations"] for c, _ in batch]),
        pa.array([m["cluster_msa"].tolist() for _, m in batch]),
        pa.array([m["cluster_has_deletion"].tolist() for _, m in batch]),
        pa.array([m["cluster_deletion_value"].tolist() for _, m in batch]),
        pa.array([m["cluster_deletion_mean"].tolist() for _, m in batch]),
        pa.array([m["cluster_profile"].tolist() for _, m in batch]),
        pa.array([m["extra_msa"].tolist() for _, m in batch]),
        pa.array([m["extra_msa_has_deletion"].tolist() for _, m in batch]),
        pa.array([m["extra_msa_deletion_value"].tolist() for _, m in batch]),
    ]


def dump_to_ipc(db_manager, msa_output_dir, output, executor, batch_size=5):
    chains = get_ready_chains(db_manager, msa_output_dir)
    msa_feat_getter = partial(get_msa_features, msa_output_dir)

    num_chains = 0
    with pa.OSFile(str(output), mode="w") as f:
        with pa.ipc.new_file(f, SCHEMA) as writer:
            for chain_batch in batched(chains, batch_size):
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        get_record_batch(executor, msa_feat_getter, chain_batch), schema=SCHEMA
                    )
                )
                num_chains += len(chain_batch)
                logging.info(f"Wrote {num_chains} chains to IPC file")
    logging.info(f"Finished writing {num_chains} chains to {output}")

import logging
import pyarrow as pa


def get_ready_chains(db_manager):
    return list(db_manager.chains().find({"msa": {"$exists": 1}}))


def dump_to_ipc(db_manager, output):
    chains = get_ready_chains(db_manager)
    table = {
        "structure_id": pa.array([c["_id"]["structure_id"] for c in chains]),
        "chain_id": pa.array([c["_id"]["chain_id"] for c in chains]),
        "rotations": pa.array([c["rotations"] for c in chains]),
        "translations": pa.array([c["translations"] for c in chains]),
        "sequence": pa.array([c["sequence"] for c in chains]),
        "positions": pa.array([c["positions"] for c in chains]),
        "msa": pa.array([c["msa"] for c in chains]),
    }
    table = pa.Table.from_arrays(list(table.values()), names=list(table.keys()))
    with pa.OSFile(str(output), mode="w") as f:
        with pa.ipc.new_file(f, table.schema) as writer:
            writer.write_table(table)
    logging.info(f"Wrote {len(chains)} chains to {output}")

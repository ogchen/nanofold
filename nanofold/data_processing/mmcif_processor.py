import glob
import os
import logging
from functools import partial
from itertools import batched

from nanofold.data_processing.mmcif_parser import get_model_id
from nanofold.data_processing.mmcif_parser import parse_mmcif_file


def list_available_mmcif(mmcif_dir):
    search_glob = os.path.join(mmcif_dir, "*.cif")
    return glob.glob(search_glob)


def get_files_to_process(db_manager, mmcif_dir):
    pdb_files = list_available_mmcif(mmcif_dir)
    processed = [r["_id"] for r in db_manager.processed_mmcif_files().find()]
    pdb_files = [f for f in pdb_files if get_model_id(f) not in processed]
    logging.info(f"Found {len(pdb_files)} files to parse, {len(processed)} processed mmCIF files")
    return pdb_files


def to_chains_document(chains):
    return [
        {
            "_id": {
                "structure_id": c["structure_id"],
                "chain_id": c["chain_id"],
            },
            "rotations": c["rotations"].tolist(),
            "translations": c["translations"].tolist(),
            "sequence": c["sequence"],
            "positions": c["positions"],
        }
        for c in chains
    ]


def process_batch(db_manager, batch, executor):
    parser = partial(parse_mmcif_file, capture_errors=True)
    result = [c for chains in executor.map(parser, batch) for c in chains]
    db_manager.chains().insert_many(to_chains_document(result))
    db_manager.processed_mmcif_files().insert_many([{"_id": get_model_id(f)} for f in batch])


def process_mmcif_files(db_manager, executor, mmcif_dir, batch_size):
    files = get_files_to_process(db_manager, mmcif_dir)

    for i, batch in enumerate(batched(files, batch_size)):
        process_batch(db_manager, batch, executor)
        logging.info(f"Parsed {i * batch_size + len(batch)}/{len(files)} files")

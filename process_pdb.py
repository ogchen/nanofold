import argparse
import pyarrow as pa
import pyarrow.compute as pc
import logging
from Bio.PDB.PDBExceptions import PDBConstructionException
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path

from nanofold.common.chain_record import ChainRecord
from nanofold.data_processing.mmcif import list_available_mmcif
from nanofold.data_processing.mmcif import load_model
from nanofold.data_processing.mmcif import parse_chains

PROCESSED_SCHEMA = pa.schema(
    [
        ("model_id", pa.string()),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", help="Batch size of files to process", default=1000, type=int
    )
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files", type=Path)
    parser.add_argument("-o", "--output", help="Directory to store output files", type=Path)
    parser.add_argument("-l", "--logging", help="Logging level", default="INFO")
    return parser.parse_args()


def process_pdb_file(filepath):
    try:
        model = load_model(filepath)
    except PDBConstructionException as e:
        logging.warning(f"Got PDB construction error for file={filepath}, error={e}")
        return []
    return parse_chains(model)


def get_id(filepath):
    return Path(filepath).stem.lower()


def load_tables(processed_ids_path, pdb_data_path):
    if processed_ids_path.exists():
        with pa.OSFile(str(processed_ids_path), mode="rb") as f:
            processed_ids = pa.ipc.open_file(f).read_all()
    else:
        processed_ids = None

    if pdb_data_path.exists():
        with pa.OSFile(str(pdb_data_path), mode="rb") as f:
            pdb_data = pa.ipc.open_file(f).read_all()
    else:
        pdb_data = None

    return processed_ids, pdb_data


def get_files_to_process(mmcif_dir, processed_ids):
    pdb_files = list_available_mmcif(mmcif_dir)
    logging.info(f"Found {len(pdb_files)} files to process")
    if processed_ids is not None:
        found_ids = processed_ids.column("model_id").to_pylist()
        pdb_files = [f for f in pdb_files if get_id(f) not in found_ids]
        logging.info(f"{len(found_ids)} files already processed, {len(pdb_files)} remaining")
    return pdb_files


def compute_pdb_data(executor, pdb_files, batch_size, pdb_data):
    batches = []
    for i, batch in enumerate(batched(pdb_files, batch_size)):
        result = [c for chains in executor.map(process_pdb_file, batch) for c in chains]
        batches.append(ChainRecord.to_record_batch(result))
        logging.info(f"Processed {i * batch_size + len(batch)}/{len(pdb_files)} files")

    new_pdb_data = pa.Table.from_batches(batches, schema=ChainRecord.SCHEMA)
    if pdb_data is not None:
        return pc.concat_tables([pdb_data, new_pdb_data]) if new_pdb_data.num_rows > 0 else pdb_data
    return new_pdb_data


def compute_processed_ids(pdb_files, processed_ids):
    new_processed_ids = pa.Table.from_arrays(
        [pa.array([get_id(f) for f in pdb_files])], schema=PROCESSED_SCHEMA
    )
    return (
        pc.concat_tables([processed_ids, new_processed_ids]) if processed_ids else new_processed_ids
    )


def write_table(filepath, table, schema):
    with pa.OSFile(str(filepath), mode="w") as f:
        with pa.ipc.new_file(f, schema) as writer:
            writer.write_table(table)


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.logging.upper()))
    processed_ids_path = args.output / "processed_ids.arrow"
    pdb_data_path = args.output / "pdb_data.arrow"

    processed_ids, pdb_data = load_tables(processed_ids_path, pdb_data_path)
    pdb_files = get_files_to_process(args.mmcif, processed_ids)

    with ProcessPoolExecutor() as executor:
        if len(pdb_files) != 0:
            pdb_data = compute_pdb_data(executor, pdb_files, args.batch, pdb_data)
            processed_ids = compute_processed_ids(pdb_files, processed_ids)
            write_table(pdb_data_path, pdb_data, ChainRecord.SCHEMA)
            write_table(processed_ids_path, processed_ids, PROCESSED_SCHEMA)
        logging.info(f"Finished processing {len(pdb_files)} files")


if __name__ == "__main__":
    main()

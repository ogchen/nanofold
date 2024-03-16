import argparse
import pyarrow as pa
import logging
from Bio.PDB.PDBExceptions import PDBConstructionException
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path

from nanofold.common.chain import Chain
from nanofold.data_processing.mmcif import list_available_mmcif
from nanofold.data_processing.mmcif import load_model
from nanofold.data_processing.mmcif import parse_chains


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", help="Batch size of files to process", default=1000, type=int
    )
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files", type=Path)
    parser.add_argument("-o", "--output", help="Output Arrow IPC file", type=Path)
    parser.add_argument("-l", "--logging", help="Logging level", default="INFO")
    return parser.parse_args()


def process_pdb_file(filepath):
    try:
        model = load_model(filepath)
    except PDBConstructionException as e:
        logging.warn(f"Got PDB construction error for file={filepath}, error={e}")
        return []
    return parse_chains(model)


def get_id(filepath):
    return Path(filepath).stem.lower()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.logging.upper()))

    pdb_files = list_available_mmcif(args.mmcif)
    logging.info(f"Found {len(pdb_files)} files to process")

    with ProcessPoolExecutor() as executor:
        with pa.OSFile(str(args.output), mode="w") as f:
            with pa.ipc.new_file(f, Chain.SCHEMA) as writer:
                for i, batch in enumerate(batched(pdb_files, args.batch)):
                    result = [c for chains in executor.map(process_pdb_file, batch) for c in chains]
                    writer.write_batch(Chain.to_record_batch(result))
                    logging.info(f"Processed {i * args.batch + len(batch)}/{len(pdb_files)} files")
    logging.info(f"Finished processing {len(pdb_files)} files, output written to {args.output}")


if __name__ == "__main__":
    main()

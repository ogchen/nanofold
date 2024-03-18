import argparse
import pyarrow as pa
import pyarrow.compute as pc
import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path

from nanofold.data_processing.chain_record import ChainRecord
from nanofold.data_processing.ipc import load_table
from nanofold.data_processing.ipc import write_table
from nanofold.data_processing.mmcif_processor import IDS_SCHEMA
from nanofold.data_processing.mmcif_processor import process_mmcif_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", help="Batch size of files to process", default=1000, type=int
    )
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files", type=Path)
    parser.add_argument("-o", "--output", help="Directory to store output files", type=Path)
    parser.add_argument("-l", "--logging", help="Logging level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.logging.upper()))
    ids_path = args.output / "processed_ids.arrow"
    pdb_data_path = args.output / "pdb_data.arrow"

    with ProcessPoolExecutor() as executor:
        data_table = load_table(pdb_data_path)
        ids_table = load_table(ids_path)
        write_tables, data_table, ids_table = process_mmcif_files(
            executor, ids_table, data_table, args.mmcif, args.batch
        )
        if write_tables:
            write_table(pdb_data_path, data_table, ChainRecord.SCHEMA)
            write_table(ids_path, ids_table, IDS_SCHEMA)


if __name__ == "__main__":
    main()

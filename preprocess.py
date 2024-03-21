import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from nanofold.data_processing.db import DBManager
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
    db_manager = DBManager(uri=os.getenv("MONGODB_URI"))

    with ProcessPoolExecutor() as executor:
        process_mmcif_files(db_manager, executor, args.mmcif, args.batch)


if __name__ == "__main__":
    main()

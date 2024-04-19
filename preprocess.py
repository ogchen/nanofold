import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nanofold.data_processing.db import DBManager
from nanofold.data_processing.hhblits import HHblitsRunner
from nanofold.data_processing.ipc import dump_to_ipc
from nanofold.data_processing.mmcif_processor import process_mmcif_files
from nanofold.data_processing.msa_builder import build_msa
from nanofold.data_processing.msa_runner import MSARunner
from nanofold.data_processing.template import build_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", help="Batch size of files to process", default=1000, type=int
    )
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files", type=Path)
    parser.add_argument("-o", "--output", help="Directory to store output files", type=Path)
    parser.add_argument("-s", "--small_bfd", help="Small BFD file", type=Path)
    parser.add_argument("-p", "--pdb70", help="PDB70 file", type=Path)
    parser.add_argument("--dump-only", help="Dump IPC file only", action="store_true")
    parser.add_argument("-l", "--logging", help="Logging level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=getattr(logging, args.logging.upper()),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    msa_output_dir = args.output / "msa"
    msa_output_dir.mkdir(exist_ok=True)
    ipc_output_path = args.output / "features.arrow"
    jackhmmer_results_path = args.output / "small_bfd_cache"
    jackhmmer_results_path.mkdir(exist_ok=True)
    template_cache_dir = args.output / "templates_cache"
    template_cache_dir.mkdir(exist_ok=True)

    db_manager = DBManager(uri=os.getenv("MONGODB_URI"))
    msa_runner = MSARunner(
        shutil.which("jackhmmer"),
        args.small_bfd,
        jackhmmer_results_path,
        num_cpus=1,
        max_sequences=500,
    )
    hhblits_runner = HHblitsRunner(
        shutil.which("hhblits"),
        args.pdb70,
        template_cache_dir,
    )

    if not args.dump_only:
        with ProcessPoolExecutor() as executor:
            process_mmcif_files(db_manager, executor, args.mmcif, args.batch)

    with ThreadPoolExecutor() as executor:
        if not args.dump_only:
            build_msa(msa_runner, db_manager, executor, msa_output_dir)
            build_template(
                hhblits_runner,
                shutil.which("reformat.pl"),
                msa_runner,
                db_manager,
                executor,
                msa_output_dir,
            )
        dump_to_ipc(db_manager, msa_output_dir, ipc_output_path, executor)


if __name__ == "__main__":
    main()

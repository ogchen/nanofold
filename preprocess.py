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
from nanofold.data_processing.msa_builder import prefetch_msa
from nanofold.data_processing.msa_builder import build_msa
from nanofold.data_processing.msa_runner import MSARunner
from nanofold.data_processing.template import build_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", help="Batch size of files to process", default=1000, type=int
    )
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files", type=Path)
    parser.add_argument("-c", "--cache", help="Directory to store cache files", type=Path)
    parser.add_argument("-o", "--output", help="Output features Arrow file", type=Path)
    parser.add_argument("-s", "--small_bfd", help="Small BFD file", type=Path)
    parser.add_argument("-p", "--pdb70", help="PDB70 database", type=Path)
    parser.add_argument("-u", "--uniclust30", help="Uniclust30 database", type=Path)
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
    msa_output_dir = args.cache / "msa"
    msa_output_dir.mkdir(exist_ok=True)
    jackhmmer_results_path = args.cache / "small_bfd_cache"
    jackhmmer_results_path.mkdir(exist_ok=True)
    uniclust30_cache_dir = args.cache / "uniclust30_cache"
    uniclust30_cache_dir.mkdir(exist_ok=True)
    template_cache_dir = args.cache / "templates_cache"
    template_cache_dir.mkdir(exist_ok=True)

    db_manager = DBManager(uri=os.getenv("MONGODB_URI"))
    small_bfd_msa_search = MSARunner(
        shutil.which("jackhmmer"),
        args.small_bfd,
        jackhmmer_results_path,
        num_cpus=1,
        max_sequences=5000,
    )
    uniclust30_msa_search = HHblitsRunner(
        shutil.which("hhblits"),
        args.uniclust30,
        uniclust30_cache_dir,
        num_iterations=3,
        num_cpu=8,
        output_format="a3m",
    )
    pdb70_template_search = HHblitsRunner(
        shutil.which("hhblits"),
        args.pdb70,
        template_cache_dir,
        num_iterations=1,
    )

    if not args.dump_only:
        with ProcessPoolExecutor() as executor:
            process_mmcif_files(db_manager, executor, args.mmcif, args.batch)

        with ThreadPoolExecutor() as executor:
            logging.info("Prefetching MSA from small BFD")
            prefetch_msa(small_bfd_msa_search, db_manager, executor, jackhmmer_results_path)

        with ThreadPoolExecutor(max_workers=2) as executor:
            logging.info("Prefetching MSA from Uniclust30")
            prefetch_msa(uniclust30_msa_search, db_manager, executor, uniclust30_cache_dir)

        with ProcessPoolExecutor(max_workers=6) as executor:
            build_msa(
                small_bfd_msa_search,
                uniclust30_msa_search,
                db_manager,
                executor,
                msa_output_dir,
                include_dirs=[jackhmmer_results_path, uniclust30_cache_dir],
            )

    with ThreadPoolExecutor() as executor:
        if not args.dump_only:
            build_template(
                pdb70_template_search,
                shutil.which("reformat.pl"),
                small_bfd_msa_search,
                db_manager,
                executor,
                msa_output_dir,
            )
        dump_to_ipc(db_manager, msa_output_dir, args.output, executor)


if __name__ == "__main__":
    main()

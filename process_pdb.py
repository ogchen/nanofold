import argparse
import csv
import logging
from itertools import batched
from pathlib import Path
from pyspark.sql import SparkSession

from nanofold.data_processing.chain import Chain
from nanofold.data_processing.mmcif import list_available_mmcif
from nanofold.data_processing.mmcif import load_model
from nanofold.data_processing.mmcif import parse_chains


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mmcif", help="Directory containing mmcif files", type=Path)
    parser.add_argument("-s", "--store", help="Directory to store processed data in", type=Path)
    parser.add_argument("-l", "--logging", help="Logging level", default="INFO")
    return parser.parse_args()


def process_pdb_file(filepath):
    model = load_model(filepath)
    chains = parse_chains(model)
    return [Chain.to_record(c) for c in chains]


def get_id(filepath):
    return Path(filepath).stem.lower()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.logging.upper()))
    id_filepath = args.store / "processed_ids.csv"
    parquet_directory = args.store / "pdb_data"
    spark = SparkSession.builder.appName("PDB Data Processing").getOrCreate()

    try:
        with open(id_filepath) as f:
            reader = csv.reader(f)
            processed_ids = list(reader)[0]
    except FileNotFoundError:
        processed_ids = []

    batch_size = 500
    pdb_files = list_available_mmcif(args.mmcif)
    pdb_files = list(filter(lambda x: get_id(x) not in processed_ids, pdb_files))
    logging.info(f"Found {len(pdb_files)} files to process")

    for i, batch in enumerate(batched(pdb_files, batch_size)):
        pdb_files_rdd = spark.sparkContext.parallelize(batch)
        data = pdb_files_rdd.flatMap(process_pdb_file)

        if not data.isEmpty():
            df = spark.createDataFrame(data)
            df.write.mode("append").parquet(str(parquet_directory))

        processed_ids += [get_id(f) for f in batch]
        with open(id_filepath, mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(processed_ids)
        logging.info(f"Processed {i * batch_size + len(batch)}/{len(pdb_files)} files")


if __name__ == "__main__":
    main()

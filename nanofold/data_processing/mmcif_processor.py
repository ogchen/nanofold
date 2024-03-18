import glob
import os
import pyarrow as pa
import pyarrow.compute as pc
import logging
from itertools import batched

from nanofold.data_processing.chain_record import ChainRecord
from nanofold.data_processing.mmcif_parser import get_model_id
from nanofold.data_processing.mmcif_parser import parse_pdb_file

IDS_SCHEMA = pa.schema(
    [
        ("model_id", pa.string()),
    ]
)


def list_available_mmcif(mmcif_dir):
    search_glob = os.path.join(mmcif_dir, "*.cif")
    return glob.glob(search_glob)


def get_processed_ids(ids_table):
    return set(ids_table.column("model_id").to_pylist()) if ids_table is not None else set()


def get_files_to_process(mmcif_dir, ids_table):
    pdb_files = list_available_mmcif(mmcif_dir)
    ids = get_processed_ids(ids_table)
    pdb_files = [f for f in pdb_files if get_model_id(f) not in ids]
    logging.info(f"Found {len(ids)} processed files, {len(pdb_files)} remaining")
    return pdb_files


def compute_record_batches(executor, pdb_files, batch_size):
    batches = []
    for i, batch in enumerate(batched(pdb_files, batch_size)):
        result = [c for chains in executor.map(parse_pdb_file, batch) for c in chains]
        batches.append(ChainRecord.to_record_batch(result))
        logging.info(f"Processed {i * batch_size + len(batch)}/{len(pdb_files)} files")
    return batches


def update_data_table(batches, data_table):
    if len(batches) == 0:
        return data_table
    new_pdb_data = pa.Table.from_batches(batches, schema=ChainRecord.SCHEMA)
    if data_table is None:
        return new_pdb_data
    return pc.concat_tables([data_table, new_pdb_data])


def update_ids_table(pdb_files, ids_table):
    if len(pdb_files) == 0:
        return ids_table
    new_ids_table = pa.Table.from_arrays(
        [pa.array([get_model_id(f) for f in pdb_files])], schema=IDS_SCHEMA
    )
    return pc.concat_tables([ids_table, new_ids_table]) if ids_table else new_ids_table


def process_mmcif_files(executor, ids_table, data_table, mmcif_dir, batch_size):
    files = get_files_to_process(mmcif_dir, ids_table)

    write_tables = False
    if len(files) != 0:
        batches = compute_record_batches(executor, files, batch_size)
        data_table = update_data_table(batches, data_table)
        ids_table = update_ids_table(files, ids_table)
        write_tables = True
    logging.info(f"Finished processing {len(files)} files")
    return write_tables, data_table, ids_table

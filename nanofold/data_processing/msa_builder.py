import glob
import gzip
import itertools
import logging
import math
import numpy as np
import os
import pickle
from functools import partial
from io import StringIO
from itertools import batched
from pathlib import Path
from scipy.sparse import coo_array
from tempfile import NamedTemporaryFile

from nanofold.common.residue_definitions import RESIDUE_INDEX
from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA_WITH_MASK
from nanofold.common.residue_definitions import UNKNOWN_RESIDUE
from nanofold.data_processing import a3m_parser
from nanofold.data_processing import sto_parser


def get_chains_to_process(db_manager, exclude_dir=None, include_dirs=None, max_chains=None):
    chains = db_manager.chains().find({}, {"_id": 1, "sequence": 1})
    exclude_ids = set()
    include_ids = None

    if exclude_dir is not None:
        search_glob = os.path.join(exclude_dir, "*.gz")
        exclude_ids = set([Path(f).stem.split(".")[0] for f in glob.glob(search_glob)])

    if include_dirs is not None:
        include_ids = [
            set([Path(f).stem.split(".")[0] for f in glob.glob(os.path.join(d, "*.gz"))])
            for d in include_dirs
        ]
        include_ids = set.intersection(*include_ids)

    for c in chains:
        if (
            f"{c['_id']['structure_id']}_{c['_id']['chain_id']}" not in exclude_ids
            and (
                include_ids is None
                or f"{c['_id']['structure_id']}_{c['_id']['chain_id']}" in include_ids
            )
            and len(c["sequence"]) >= 32
        ):
            yield c
            if max_chains is not None:
                max_chains -= 1
                if max_chains <= 0:
                    break


def execute_msa_search(msa_runner, chain):
    with NamedTemporaryFile(mode="w") as tmp:
        id = f"{chain['_id']['structure_id'].lower()}_{chain['_id']['chain_id']}"
        fasta = f">{id}\n{chain['sequence']}"
        tmp.writelines(fasta)
        tmp.flush()
        return msa_runner.run(tmp.name, id)


def encode_one_hot_alignments(alignments):
    indices = np.array(
        [
            [
                RESIDUE_INDEX_MSA_WITH_MASK.get(r, RESIDUE_INDEX_MSA_WITH_MASK[UNKNOWN_RESIDUE[0]])
                for r in a
            ]
            for a in alignments
        ]
    )
    return np.eye(len(RESIDUE_INDEX_MSA_WITH_MASK))[indices]


def normalize_to_unit(x):
    return 2 / math.pi * np.arctan(x / 3)


def preprocess_msa(alignments, deletion_matrix):
    indices = np.arange(len(alignments))
    np.random.shuffle(indices[1:])
    alignments_one_hot = encode_one_hot_alignments([alignments[i] for i in indices])
    deletion_matrix = np.array([deletion_matrix[i] for i in indices])
    return alignments_one_hot, deletion_matrix


def parse_msa_features(alignments, deletion_matrix, num_seq):
    alignments_one_hot, deletion_matrix = preprocess_msa(alignments, deletion_matrix)
    return {
        "msa": alignments_one_hot[:num_seq],
        "has_deletion": (deletion_matrix[:num_seq] > 0),
        "deletion_value": normalize_to_unit(deletion_matrix[:num_seq]),
    }, {
        "profile": np.mean(alignments_one_hot, axis=0),
        "deletion_mean": np.mean(deletion_matrix, axis=0),
    }


def to_sparse_features(msa_feat):
    result = {}
    for k, v in msa_feat.items():
        square_arr = np.moveaxis(v, 1, 0).reshape(v.shape[1], -1)
        sparse_arr = coo_array(square_arr)
        result[f"{k}_shape"] = square_arr.shape
        result[f"{k}_data"] = sparse_arr.data.tolist()
        result[f"{k}_coords"] = [c.tolist() for c in sparse_arr.coords]
    return result


def get_msa(uniclust30_msa_search, small_bfd_msa_search, chain, num_seq=4096):
    small_bfd_result = execute_msa_search(small_bfd_msa_search, chain)
    small_bfd_alignments, small_bfd_deletion_matrix = sto_parser.extract_alignments(
        small_bfd_result()
    )
    uniclust30_result = execute_msa_search(uniclust30_msa_search, chain)
    uniclust30_alignments, uniclust30_deletion_matrix = a3m_parser.extract_alignments(
        uniclust30_result()
    )

    if uniclust30_alignments[0] != small_bfd_alignments[0]:
        if small_bfd_alignments[0].startswith(uniclust30_alignments[0]):
            small_bfd_alignments = [
                s[: len(uniclust30_alignments[0])] for s in small_bfd_alignments
            ]
            small_bfd_deletion_matrix = [
                d[: len(uniclust30_deletion_matrix[0])] for d in small_bfd_deletion_matrix
            ]
        else:
            logging.warn(
                f"Query sequence for small_bfd and uniclust30 do not match for {chain['_id']}, using only uniclust30 MSA"
            )
            small_bfd_alignments = []
            small_bfd_deletion_matrix = []

    alignments = uniclust30_alignments
    deletion_matrix = uniclust30_deletion_matrix
    for s, d in zip(small_bfd_alignments[1:], small_bfd_deletion_matrix[1:]):
        if s not in alignments:
            alignments.append(s)
            deletion_matrix.append(d)

    if alignments[0] != chain["sequence"]:
        raise ValueError("Query sequence does not match the first sequence in the MSA")
    msa_features, profile_features = parse_msa_features(alignments, deletion_matrix, num_seq)
    return {**to_sparse_features(msa_features), **profile_features}


def prefetch_msa(msa_runner, db_manager, executor, output_dir, batch_size=50):
    chains = get_chains_to_process(db_manager, output_dir)
    prefetch = partial(execute_msa_search, msa_runner)

    for j, chain_batch in enumerate(batched(chains, batch_size)):
        result = executor.map(prefetch, chain_batch)
        for i, c in enumerate(chain_batch):
            try:
                next(result)
            except Exception as e:
                logging.error(f"Failure while prefetching MSA for chain {c['_id']}: {repr(e)}")
                continue
            logging.info(f"Prefetched raw MSA for {j * batch_size + i} chains")


def build_msa(
    small_bfd_msa_search,
    uniclust30_msa_search,
    db_manager,
    executor,
    msa_output_dir,
    include_dirs,
    batch_size=50,
):
    chains = get_chains_to_process(
        db_manager, exclude_dir=msa_output_dir, include_dirs=include_dirs
    )

    get_msa_func = partial(get_msa, uniclust30_msa_search, small_bfd_msa_search)

    for j, chain_batch in enumerate(batched(chains, batch_size)):
        chain_batch_small = [c for c in chain_batch if len(c["sequence"]) < 600]
        chain_batch_large = [c for c in chain_batch if c not in chain_batch_small]
        chain_batch = itertools.chain(chain_batch_small, chain_batch_large)
        result = itertools.chain(
            executor.map(get_msa_func, chain_batch_small),
            (get_msa_func(c) for c in chain_batch_large),
        )
        i = 0
        while True:
            try:
                c = next(chain_batch)
                features = next(result)
                with gzip.open(
                    msa_output_dir / f"{c['_id']['structure_id']}_{c['_id']['chain_id']}.pkl.gz",
                    "wb",
                ) as f:
                    pickle.dump(features, f)
            except StopIteration:
                break
            except Exception as e:
                logging.error(
                    f"Failure fetching alignment contents for chain {c['_id']}: {repr(e)}"
                )
                continue
            logging.info(f"Fetched MSA alignments for {j * batch_size + i} chains")
            i += 1

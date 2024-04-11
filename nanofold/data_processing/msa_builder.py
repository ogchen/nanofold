import logging
import math
import numpy as np
import pickle
from functools import partial
from io import StringIO
from itertools import batched
from tempfile import NamedTemporaryFile

from nanofold.data_processing.sto_parser import parse_msa
from nanofold.common.residue_definitions import RESIDUE_INDEX
from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA_WITH_MASK
from nanofold.common.residue_definitions import UNKNOWN_RESIDUE


def get_chains_to_process(db_manager):
    chains = db_manager.chains().find({"msa_feat": {"$exists": 0}}, {"_id": 1, "sequence": 1})
    return list(chains)


def get_msa(msa_runner, chain):
    with NamedTemporaryFile(mode="w") as tmp:
        id = f"{chain['_id']['structure_id'].lower()}_{chain['_id']['chain_id']}"
        fasta = f">{id}\n{chain['sequence']}"
        tmp.writelines(fasta)
        tmp.flush()
        return msa_runner.run(tmp.name, id)


def get_sto_contents(msa_runner, executor, chains, batch_size=100):
    get_result = partial(get_msa, msa_runner)
    for i, batch in enumerate(batched(chains, batch_size)):
        result = executor.map(get_result, batch)
        logging.info(
            f"Fetched small BFD alignments for {i * batch_size + len(batch)}/{len(chains)} chains"
        )
        for chain in batch:
            try:
                yield chain, next(result)
            except Exception as e:
                logging.error(f"Failure fetching alignment contents for chain {chain['_id']}: {e}")
                continue


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


def get_msa_feat(alignments, deletion_matrix, num_msa_clusters, batch_size=500):
    cluster_msa = alignments[:num_msa_clusters]
    cluster_index = np.argmax(cluster_msa[..., : len(RESIDUE_INDEX)], axis=-1)[:, np.newaxis, :]
    cluster_profile = np.copy(cluster_msa)
    cluster_deletion_mean = np.copy(deletion_matrix[:num_msa_clusters])
    cluster_counts = np.ones(len(cluster_msa))

    for i in range(num_msa_clusters, len(alignments), batch_size):
        batch_alignments = alignments[i : i + batch_size]
        batch_deletion_matrix = deletion_matrix[i : i + batch_size]
        difference = (
            cluster_index
            - np.argmax(batch_alignments[..., : len(RESIDUE_INDEX)], axis=-1)[np.newaxis, :, :]
        )
        closest_cluster = np.argmin(np.sum(difference == 0, axis=-1), axis=0)
        for j, k in enumerate(closest_cluster):
            cluster_profile[k] += batch_alignments[j]
            cluster_counts[k] += 1
            cluster_deletion_mean[k] += batch_deletion_matrix[j]

    cluster_has_deletion = deletion_matrix[:num_msa_clusters] > 0
    cluster_deletion_value = normalize_to_unit(deletion_matrix[:num_msa_clusters])
    cluster_deletion_mean = normalize_to_unit(cluster_deletion_mean / cluster_counts[:, np.newaxis])
    cluster_profile = cluster_profile / cluster_counts[:, np.newaxis, np.newaxis]

    return cluster_msa, np.concatenate(
        [
            cluster_has_deletion[..., np.newaxis],
            cluster_deletion_value[..., np.newaxis],
            cluster_deletion_mean[..., np.newaxis],
            cluster_profile,
        ],
        axis=-1,
    )


def preprocess_msa(alignments, deletion_matrix):
    indices = np.arange(1, len(alignments))
    np.random.shuffle(indices)
    alignments_one_hot = encode_one_hot_alignments(
        [alignments[0], *[alignments[i] for i in indices]]
    )
    deletion_matrix = np.array([deletion_matrix[0], *[deletion_matrix[i] for i in indices]])
    return alignments_one_hot, deletion_matrix


def get_extra_msa_seq(alignments_one_hot, deletion_matrix, num_msa_clusters, num_extra_seq):
    indices = np.arange(num_msa_clusters, len(alignments_one_hot))
    np.random.shuffle(indices)
    indices = indices[:num_extra_seq]
    extra_msa_has_deletion = deletion_matrix[indices] > 0
    extra_msa_deletion_value = normalize_to_unit(deletion_matrix[indices])

    return np.concatenate(
        [
            alignments_one_hot[indices],
            extra_msa_has_deletion[..., np.newaxis],
            extra_msa_deletion_value[..., np.newaxis],
        ],
        axis=-1,
    )


def parse_msa_features(alignments, deletion_matrix, num_msa_clusters=64, num_extra_seq=196):
    alignments_one_hot, deletion_matrix = preprocess_msa(alignments, deletion_matrix)
    cluster_msa, msa_feat = get_msa_feat(alignments_one_hot, deletion_matrix, num_msa_clusters)
    extra_msa_feat = get_extra_msa_seq(
        alignments_one_hot, deletion_matrix, len(cluster_msa), num_extra_seq
    )
    return {
        "cluster_msa": cluster_msa,
        "msa_feat": msa_feat,
        "extra_msa_feat": extra_msa_feat,
    }


def build_msa(msa_runner, db_manager, executor, msa_output_dir):
    chains = get_chains_to_process(db_manager)
    total_num_chains = db_manager.chains().count_documents({})
    logging.info(f"Found {len(chains)}/{total_num_chains} chains missing MSA")
    for chain, sto_contents in get_sto_contents(msa_runner, executor, chains):
        try:
            alignments, deletion_matrix = parse_msa(StringIO(sto_contents))
            if alignments[0] != chain["sequence"]:
                logging.error(f"Chain {chain['_id']} has a mismatching sequence and alignment")
                continue
            features = parse_msa_features(alignments, deletion_matrix)
            with open(
                msa_output_dir / f"{chain['_id']['structure_id']}_{chain['_id']['chain_id']}.pkl",
                "wb",
            ) as f:
                pickle.dump(features, f)
            db_manager.chains().update_one({"_id": chain["_id"]}, {"$set": {"msa_feat": True}})
        except Exception as e:
            logging.error(f"Failed to build msa for chain {chain['_id']}: {e}")

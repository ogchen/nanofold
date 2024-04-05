from torch.utils.data import IterableDataset
import logging
import math
import numpy as np
import polars as pl
import random
import torch

from nanofold.common.residue_definitions import get_atom_positions
from nanofold.common.residue_definitions import MSA_GAP
from nanofold.common.residue_definitions import RESIDUE_INDEX
from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA
from nanofold.common.residue_definitions import UNKNOWN_RESIDUE


SAMPLE_SIZE = 100


def accept_chain(row):
    return np.random.rand() < max(min(row["length"], 512), 256) / 512


def encode_one_hot(seq):
    indices = torch.tensor([RESIDUE_INDEX[residue] for residue in seq])
    return torch.nn.functional.one_hot(indices, num_classes=len(RESIDUE_INDEX)).float()


def encode_one_hot_alignments(alignments):
    indices = torch.tensor(
        [
            [RESIDUE_INDEX_MSA.get(r, RESIDUE_INDEX_MSA[UNKNOWN_RESIDUE[0]]) for r in a]
            for a in alignments
        ]
    )
    return torch.nn.functional.one_hot(indices, num_classes=len(RESIDUE_INDEX_MSA)).float()


def encode_deletion_matrix(deletion_matrix):
    counts = torch.tensor(deletion_matrix, dtype=torch.float32)
    has_deletion = counts > 0
    deletion_value = 2 / math.pi * torch.arctan(counts / 3)
    return torch.cat((has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)), dim=-1)


def encode_msa(msa):
    alignments_one_hot = encode_one_hot_alignments([m[0] for m in msa])
    deletion_feat = encode_deletion_matrix([m[1] for m in msa])
    return torch.cat((alignments_one_hot, deletion_feat), dim=-1)


def preprocess_msa(msa, num_msa):
    msa = [m for m in msa if not all(c == MSA_GAP for c in m[0])]
    query = msa[0]

    # Deduplicate MSA
    hashable_msa = [(a, tuple(d)) for a, d in msa[1:]]
    deduplicated = [(a, list(d)) for a, d in set(hashable_msa)]

    # Shuffle MSA
    random.shuffle(deduplicated)

    # Pad or truncate MSA
    msa = [query, *[m for m in deduplicated if m[0] != query]][:num_msa]
    for _ in range(num_msa - len(msa)):
        msa.append((MSA_GAP * len(query[0]), [0] * len(query[1])))

    return msa


class ChainDataset(IterableDataset):
    def __init__(self, df, indices, residue_crop_size, num_msa):
        super().__init__()
        self.residue_crop_size = residue_crop_size
        self.num_msa = num_msa
        self.df = df.with_row_count("index").with_columns(length=pl.col("sequence").str.len_chars())
        self.indices = indices

    @classmethod
    def construct_datasets(cls, features_file, train_split, *args, **kwargs):
        df = pl.read_ipc(
            features_file, columns=["rotations", "translations", "sequence", "positions", "msa"]
        )
        logging.info(f"Dataframe loaded, estimated size {df.estimated_size(unit='mb'):.2f} MB")
        train_size = int(train_split * len(df))
        if train_size <= 0 or train_split >= len(df):
            raise ValueError(f"train_size must be between 0 and len(df), got {train_size}")
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        return cls(df, indices[:train_size], *args, **kwargs), cls(
            df, indices[train_size:], *args, **kwargs
        )

    def __iter__(self):
        while True:
            sampled_indices = np.random.choice(self.indices, SAMPLE_SIZE)
            sample = self.df.filter(pl.col("index").is_in(sampled_indices)).filter(
                pl.col("length") >= self.residue_crop_size
            )
            sample = sample.with_columns(
                start=pl.lit(np.random.randint(sample["length"] - self.residue_crop_size + 1)),
            ).with_columns(
                positions=pl.col("positions").list.slice(pl.col("start"), self.residue_crop_size),
                sequence=pl.col("sequence").str.slice(pl.col("start"), self.residue_crop_size),
                translations=pl.col("translations").list.slice(
                    pl.col("start"), self.residue_crop_size
                ),
                rotations=pl.col("rotations").list.slice(pl.col("start"), self.residue_crop_size),
            )

            for row in sample.iter_rows(named=True):
                if accept_chain(row):
                    yield self.parse_features(row)

    def parse_msa_features(self, row):
        msa = [
            (
                a[row["start"] : row["start"] + self.residue_crop_size],
                d[row["start"] : row["start"] + self.residue_crop_size],
            )
            for a, d in zip(row["msa"]["alignments"], row["msa"]["deletion_matrix"])
        ]
        return encode_msa(preprocess_msa(msa, self.num_msa))

    def parse_features(self, row):
        features = {
            "rotations": torch.tensor(row["rotations"]),
            "translations": torch.tensor(row["translations"]),
            "local_coords": torch.tensor(
                [[p[1] for p in get_atom_positions(r)] for r in row["sequence"]]
            ),
            "target_feat": encode_one_hot(row["sequence"]),
            "msa_feat": self.parse_msa_features(row),
            "positions": torch.tensor(row["positions"]),
        }
        return features

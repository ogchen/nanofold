from torch.utils.data import IterableDataset
import logging
import math
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import random
import torch

from nanofold.common.residue_definitions import get_atom_positions
from nanofold.common.residue_definitions import MSA_GAP
from nanofold.common.residue_definitions import MSA_MASK_TOKEN
from nanofold.common.residue_definitions import RESIDUE_INDEX
from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA_WITH_MASK
from nanofold.common.residue_definitions import UNKNOWN_RESIDUE


SAMPLE_SIZE = 100


def accept_chain(row):
    return np.random.rand() < max(min(row.length, 512), 256) / 512


def encode_one_hot(seq):
    indices = torch.tensor([RESIDUE_INDEX[residue] for residue in seq])
    return torch.nn.functional.one_hot(indices, num_classes=len(RESIDUE_INDEX)).float()


def encode_one_hot_alignments(alignments):
    indices = torch.tensor(
        [
            [
                RESIDUE_INDEX_MSA_WITH_MASK.get(r, RESIDUE_INDEX_MSA_WITH_MASK[UNKNOWN_RESIDUE[0]])
                for r in a
            ]
            for a in alignments
        ]
    )
    return torch.nn.functional.one_hot(
        indices, num_classes=len(RESIDUE_INDEX_MSA_WITH_MASK)
    ).float()


def encode_deletion_matrix(deletion_matrix):
    counts = torch.stack(deletion_matrix)
    has_deletion = counts > 0
    deletion_value = 2 / math.pi * torch.arctan(counts / 3)
    return torch.cat((has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)), dim=-1)


def encode_msa(msa):
    alignments_one_hot = encode_one_hot_alignments([m[0] for m in msa])
    deletion_feat = encode_deletion_matrix([m[1] for m in msa])
    return alignments_one_hot, deletion_feat


def preprocess_msa(msa, num_msa):
    msa = zip(*msa.values())
    msa = [m for m in msa if not all(c == MSA_GAP for c in m[0])]
    query = (msa[0][0], torch.tensor(msa[0][1]))

    # Deduplicate MSA
    deduplicated = list(set([(a, torch.tensor(d)) for a, d in msa[1:]]))

    # Shuffle MSA
    random.shuffle(deduplicated)

    # Pad or truncate MSA
    msa = [query, *[m for m in deduplicated if m[0] != query]][:num_msa]
    for _ in range(num_msa - len(msa)):
        msa.append((MSA_GAP * len(query[0]), torch.zeros(len(query[1]), dtype=torch.int32)))

    return msa


class ChainDataset(IterableDataset):
    def __init__(self, table, indices, residue_crop_size, num_msa):
        super().__init__()
        self.residue_crop_size = residue_crop_size
        self.num_msa = num_msa
        self.table = table
        self.indices = indices

    @classmethod
    def construct_datasets(cls, features_file, train_split, *args, **kwargs):
        with pa.memory_map(str(features_file)) as source:
            with pa.ipc.open_file(source) as reader:
                table = reader.read_all()
        table_size = table.get_total_buffer_size() / (1024**2)
        logging.info(f"Features table loaded, size {table_size:.2f} MB")
        train_size = int(train_split * table.num_rows)
        if train_size <= 0 or train_size > table.num_rows:
            raise ValueError(f"train_size must be between 0 and {table.num_rows}, got {train_size}")
        indices = np.arange(table.num_rows)
        np.random.shuffle(indices)
        table = table.append_column(
            "index",
            [np.arange(table.num_rows)],
        ).append_column(
            "length",
            pc.list_value_length(table["positions"]),
        )
        return cls(table, indices[:train_size], *args, **kwargs), cls(
            table, indices[train_size:], *args, **kwargs
        )

    def __iter__(self):
        slice_column = lambda col_name: lambda r: r[col_name][
            r["start"] : r["start"] + self.residue_crop_size
        ]
        slice_column_msa = lambda r: {
            "alignments": [
                a[r["start"] : r["start"] + self.residue_crop_size] for a in r["msa"]["alignments"]
            ],
            "deletion_matrix": [
                d[r["start"] : r["start"] + self.residue_crop_size]
                for d in r["msa"]["deletion_matrix"]
            ],
        }

        while True:
            sampled_indices = np.random.choice(self.indices, SAMPLE_SIZE)
            expression = pc.field("index").isin(sampled_indices) & (
                pc.field("length") >= self.residue_crop_size
            )
            sample = self.table.filter(expression)
            df = sample.to_pandas(use_threads=False)
            df["start"] = df["length"].apply(
                lambda x: np.random.randint(x - self.residue_crop_size + 1)
            )
            for col_name in ["positions", "sequence", "translations", "rotations"]:
                df[col_name] = df.apply(slice_column(col_name), axis=1)
            df["msa"] = df.apply(slice_column_msa, axis=1)

            for row in df.itertuples():
                if accept_chain(row):
                    yield self.parse_features(row)

    def parse_msa_features(self, msa):
        msa = preprocess_msa(msa, self.num_msa)
        alignments_one_hot, deletion_feat = encode_msa(msa)
        msa_mask = torch.rand(alignments_one_hot.shape[:-1]) < 0.1
        masked_msa_truth = msa_mask.unsqueeze(-1) * alignments_one_hot
        masked_alignments = (
            alignments_one_hot
            - masked_msa_truth
            + (msa_mask.unsqueeze(-1) * encode_one_hot_alignments(MSA_MASK_TOKEN))
        )
        return {
            "msa_feat": torch.cat((masked_alignments, deletion_feat), dim=-1),
            "msa_mask": msa_mask,
            "masked_msa_truth": masked_msa_truth,
        }

    def parse_features(self, row):
        features = {
            "rotations": torch.from_numpy(
                np.stack(np.vstack(row.rotations.tolist()).tolist())
            ).float(),
            "translations": torch.from_numpy(np.vstack(row.translations.tolist())).float(),
            "local_coords": torch.tensor(
                [[p[1] for p in get_atom_positions(r)] for r in row.sequence]
            ),
            "target_feat": encode_one_hot(row.sequence),
            "positions": torch.from_numpy(row.positions),
            **self.parse_msa_features(row.msa),
        }
        return features

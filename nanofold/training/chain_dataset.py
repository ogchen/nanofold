import logging
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from nanofold.common.residue_definitions import get_atom_positions
from nanofold.common.residue_definitions import MSA_GAP
from nanofold.common.residue_definitions import MSA_MASK_TOKEN
from nanofold.common.residue_definitions import RESIDUE_INDEX
from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA_WITH_MASK


SAMPLE_SIZE = 5


def accept_chain(row):
    return np.random.rand() < max(min(row.length, 512), 256) / 512


def encode_one_hot(seq):
    indices = torch.tensor([RESIDUE_INDEX[residue] for residue in seq])
    return torch.nn.functional.one_hot(indices, num_classes=len(RESIDUE_INDEX)).float()


class ChainDataset(IterableDataset):
    def __init__(self, table, indices, residue_crop_size, num_msa_clusters, num_extra_msa):
        super().__init__()
        self.residue_crop_size = residue_crop_size
        self.num_msa_clusters = num_msa_clusters
        self.num_extra_msa = num_extra_msa
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

        while True:
            sampled_indices = np.random.choice(self.indices, SAMPLE_SIZE)
            expression = pc.field("index").isin(sampled_indices) & (
                pc.field("length") >= self.residue_crop_size
            )
            sample = self.table.filter(expression)
            if len(sample) == 0:
                continue

            df = sample.to_pandas(use_threads=False)
            df["start"] = df["length"].apply(
                lambda x: np.random.randint(x - self.residue_crop_size + 1)
            )
            for col_name in ["positions", "sequence", "translations", "rotations"]:
                df[col_name] = df.apply(slice_column(col_name), axis=1)
            df["cluster_msa"] = df.apply(
                lambda r: [
                    s[r["start"] : r["start"] + self.residue_crop_size]
                    for s in r["cluster_msa"][: self.num_msa_clusters]
                ],
                axis=1,
            )
            df["msa_feat"] = df.apply(
                lambda r: [
                    s[r["start"] : r["start"] + self.residue_crop_size]
                    for s in r["msa_feat"][: self.num_msa_clusters]
                ],
                axis=1,
            )
            df["extra_msa_feat"] = df.apply(
                lambda r: [
                    s[r["start"] : r["start"] + self.residue_crop_size]
                    for s in r["extra_msa_feat"][: self.num_extra_msa]
                ],
                axis=1,
            )

            for row in df.itertuples():
                if accept_chain(row):
                    yield self.parse_features(row)

    def parse_msa_features(self, row):
        msa_feat = torch.from_numpy(np.stack(np.stack(row.msa_feat).tolist()))
        cluster_msa = torch.from_numpy(np.stack(np.stack(row.cluster_msa).tolist()))

        cluster_mask = torch.rand(cluster_msa.shape[:-1]) < 0.15
        replace_mask = F.one_hot(
            RESIDUE_INDEX_MSA_WITH_MASK[MSA_MASK_TOKEN]
            * torch.ones(cluster_mask.shape, dtype=torch.long),
            num_classes=len(RESIDUE_INDEX_MSA_WITH_MASK),
        )
        replace_uniform = F.one_hot(
            torch.randint(len(RESIDUE_INDEX), cluster_mask.shape),
            num_classes=len(RESIDUE_INDEX_MSA_WITH_MASK),
        )
        replace_sampled = F.one_hot(
            torch.multinomial(
                msa_feat[..., 3:].reshape(-1, len(RESIDUE_INDEX_MSA_WITH_MASK)), 1, replacement=True
            ).reshape(msa_feat.shape[:-1]),
            num_classes=len(RESIDUE_INDEX_MSA_WITH_MASK),
        )
        replace_p = torch.rand(cluster_mask.shape).unsqueeze(-1)
        replace_value = (
            (replace_p < 0.1) * replace_uniform
            + ((replace_p >= 0.1) & (replace_p < 0.2)) * replace_sampled
            + ((replace_p >= 0.2) & (replace_p < 0.9)) * replace_mask
            + (replace_p >= 0.9) * cluster_msa
        )

        masked_msa_truth = cluster_mask.unsqueeze(-1) * cluster_msa
        replaced_cluster_msa = cluster_msa - masked_msa_truth + replace_value

        extra_msa_feat = (
            torch.from_numpy(np.stack(np.stack(row.extra_msa_feat).tolist()))
            if len(row.extra_msa_feat) > 0
            else torch.empty([0, self.residue_crop_size, len(RESIDUE_INDEX_MSA_WITH_MASK) + 2])
        )

        # Padding
        padding_msa = torch.cat([cluster_msa, msa_feat[..., :2]], dim=-1)[
            : max(self.num_extra_msa - len(extra_msa_feat), 0)
        ]
        extra_msa_feat = torch.cat([extra_msa_feat, padding_msa], dim=0)
        extra_msa_feat = torch.cat(
            [
                extra_msa_feat,
                torch.ones(
                    max(self.num_extra_msa - len(extra_msa_feat), 0), self.residue_crop_size, 1
                )
                * F.one_hot(
                    torch.tensor(RESIDUE_INDEX_MSA_WITH_MASK[MSA_GAP]),
                    num_classes=len(RESIDUE_INDEX_MSA_WITH_MASK) + 2,
                ),
            ],
            dim=0,
        )

        missing_clusters = max(self.num_msa_clusters - len(replaced_cluster_msa), 0)
        replaced_cluster_msa = torch.cat(
            [
                replaced_cluster_msa,
                torch.ones(missing_clusters, self.residue_crop_size, 1)
                * F.one_hot(
                    torch.tensor(RESIDUE_INDEX_MSA_WITH_MASK[MSA_GAP]),
                    num_classes=len(RESIDUE_INDEX_MSA_WITH_MASK),
                ),
            ],
            dim=0,
        )
        msa_feat = torch.cat([msa_feat, torch.zeros(missing_clusters, *msa_feat.shape[1:])], dim=0)
        cluster_mask = torch.cat(
            [cluster_mask, torch.zeros(missing_clusters, self.residue_crop_size, dtype=torch.bool)],
            dim=0,
        )
        masked_msa_truth = torch.cat(
            [masked_msa_truth, torch.zeros((missing_clusters, *masked_msa_truth.shape[1:]))],
            dim=0,
        )

        return {
            "cluster_mask": cluster_mask,
            "masked_msa_truth": masked_msa_truth,
            "msa_feat": torch.cat([replaced_cluster_msa, msa_feat], dim=-1),
            "extra_msa_feat": extra_msa_feat,
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
            **self.parse_msa_features(row),
        }
        return features

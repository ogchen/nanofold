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
from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA


SAMPLE_SIZE = 2


def accept_chain(row):
    return np.random.rand() < max(min(row.length, 512), 256) / 512


def encode_one_hot(seq):
    indices = torch.tensor([RESIDUE_INDEX[residue] for residue in seq])
    return F.one_hot(indices, num_classes=len(RESIDUE_INDEX)).float()


def mask_replace_msa(cluster_msa, cluster_profile):
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
        torch.multinomial(cluster_profile.flatten(end_dim=-2), 1, replacement=True).reshape(
            cluster_profile.shape[:-1]
        ),
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
    return replaced_cluster_msa, masked_msa_truth, cluster_mask


def construct_gap_padding(shape):
    return torch.ones(shape).unsqueeze(-1) * F.one_hot(
        torch.tensor(RESIDUE_INDEX_MSA_WITH_MASK[MSA_GAP]),
        num_classes=len(RESIDUE_INDEX_MSA_WITH_MASK),
    )


def construct_extra_msa_feat(
    extra_msa,
    extra_msa_has_deletion,
    extra_msa_deletion_value,
    cluster_centre_msa_feat,
    num_extra_msa,
):
    if len(extra_msa) > 0:
        extra_msa = torch.from_numpy(np.stack(np.stack(extra_msa).tolist()))
        extra_msa_has_deletion = torch.from_numpy(np.stack(extra_msa_has_deletion))
        extra_msa_deletion_value = torch.from_numpy(np.stack(extra_msa_deletion_value))
        extra_msa_feat = torch.cat(
            [
                extra_msa,
                extra_msa_has_deletion.unsqueeze(-1),
                extra_msa_deletion_value.unsqueeze(-1),
            ],
            dim=-1,
        )
        extra_msa_feat = torch.cat([extra_msa_feat, cluster_centre_msa_feat], dim=0)[:num_extra_msa]
    else:
        extra_msa_feat = cluster_centre_msa_feat[:num_extra_msa]

    padding_shape = (max(num_extra_msa - len(extra_msa_feat), 0), *extra_msa_feat.shape[1:-1])
    padding = torch.cat(
        [
            construct_gap_padding(padding_shape),
            torch.zeros(*padding_shape, 2),
        ],
        dim=-1,
    )
    return torch.cat([extra_msa_feat, padding], dim=0)


def pad_msa_feat(msa_feat, masked_msa_truth, cluster_mask, num_msa_clusters):
    padding_shape = (max(num_msa_clusters - len(msa_feat), 0), *msa_feat.shape[1:-1])
    gap_padding = construct_gap_padding(padding_shape)
    padding = torch.cat([gap_padding, torch.zeros(*padding_shape, 3), gap_padding], dim=-1)
    msa_feat = torch.cat([msa_feat, padding], dim=0)
    masked_msa_truth = torch.cat(
        [masked_msa_truth, torch.zeros(*padding_shape, masked_msa_truth.shape[-1])],
        dim=0,
    )
    cluster_mask = torch.cat(
        [cluster_mask, torch.zeros(*padding_shape)],
        dim=0,
    )
    return msa_feat, masked_msa_truth, cluster_mask


class ChainDataset(IterableDataset):
    def __init__(self, table, indices, residue_crop_size, num_msa_clusters, num_extra_msa):
        super().__init__()
        self.residue_crop_size = residue_crop_size
        self.num_msa_clusters = num_msa_clusters
        self.num_extra_msa = num_extra_msa
        self.table = table
        self.indices = indices
        self.distogram_max = 50.75
        self.distogram_bins = torch.arange(3.875, self.distogram_max, 1.25)

    @classmethod
    def construct_datasets(cls, features_file, train_split, *args, **kwargs):
        with pa.memory_map(str(features_file)) as source:
            with pa.ipc.open_file(source) as reader:
                table = reader.read_all()
        table_size = table.get_total_buffer_size() / (1024**3)
        logging.info(f"Features table loaded, size {table_size:.2f} GB")
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
        slice_column_msa = lambda col_name, slice_size: lambda r: [
            s[r["start"] : r["start"] + self.residue_crop_size] for s in r[col_name][:slice_size]
        ]
        slice_column_template = lambda col_name: lambda r: [
            t[r["start"] : r["start"] + self.residue_crop_size] for t in r[col_name]
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
            for col_name in [
                "cluster_msa",
                "cluster_has_deletion",
                "cluster_deletion_value",
                "cluster_deletion_mean",
                "cluster_profile",
            ]:
                df[col_name] = df.apply(slice_column_msa(col_name, self.num_msa_clusters), axis=1)
            for col_name in ["extra_msa", "extra_msa_has_deletion", "extra_msa_deletion_value"]:
                df[col_name] = df.apply(slice_column_msa(col_name, self.num_extra_msa), axis=1)
            for col_name in ["template_mask", "template_sequence", "template_translations"]:
                df[col_name] = df.apply(slice_column_template(col_name), axis=1)

            for row in df.itertuples():
                if accept_chain(row):
                    yield self.parse_features(row)

    def parse_msa_features(self, row):
        cluster_msa = torch.from_numpy(np.stack(np.stack(row.cluster_msa).tolist())).long()
        cluster_profile = torch.from_numpy(np.stack(np.stack(row.cluster_profile).tolist()))
        cluster_has_deletion = (
            torch.from_numpy(np.stack(row.cluster_has_deletion)).long().unsqueeze(-1)
        )
        cluster_deletion_value = torch.from_numpy(np.stack(row.cluster_deletion_value)).unsqueeze(
            -1
        )
        cluster_deletion_mean = torch.from_numpy(np.stack(row.cluster_deletion_mean)).unsqueeze(-1)
        replaced_cluster_msa, masked_msa_truth, cluster_mask = mask_replace_msa(
            cluster_msa, cluster_profile
        )

        extra_msa_feat = construct_extra_msa_feat(
            row.extra_msa,
            row.extra_msa_has_deletion,
            row.extra_msa_deletion_value,
            cluster_centre_msa_feat=torch.cat(
                [cluster_msa, cluster_has_deletion, cluster_deletion_value], dim=-1
            ),
            num_extra_msa=self.num_extra_msa,
        )

        msa_feat = torch.cat(
            [
                replaced_cluster_msa,
                cluster_has_deletion,
                cluster_deletion_value,
                cluster_deletion_mean,
                cluster_profile,
            ],
            dim=-1,
        )
        msa_feat, masked_msa_truth, cluster_mask = pad_msa_feat(
            msa_feat, masked_msa_truth, cluster_mask, self.num_msa_clusters
        )
        return {
            "cluster_mask": cluster_mask,
            "masked_msa_truth": masked_msa_truth,
            "msa_feat": msa_feat,
            "extra_msa_feat": extra_msa_feat,
        }

    def parse_template_features(self, row):
        logging.info(f"Template sequence len {len(row.template_sequence)}")
        if len(row.template_sequence) == 0:
            return {
                "template_pair_feat": torch.empty(
                    (0, self.residue_crop_size, self.residue_crop_size, 84)
                ),
            }
        template_mask = torch.from_numpy(np.stack(row.template_mask))
        template_translations = torch.from_numpy(
            np.stack(np.stack(row.template_translations).tolist())
        )
        aatype_index = torch.tensor(
            [[RESIDUE_INDEX_MSA[residue] for residue in seq] for seq in row.template_sequence]
        )
        template_aatype = F.one_hot(aatype_index, num_classes=len(RESIDUE_INDEX_MSA))
        num_res = template_aatype.shape[1]
        template_aatype_pair = torch.cat(
            [
                torch.tile(template_aatype, (num_res, 1)).view(
                    -1, num_res, num_res, len(RESIDUE_INDEX_MSA)
                ),
                torch.tile(template_aatype, (1, num_res)).view(
                    -1, num_res, num_res, len(RESIDUE_INDEX_MSA)
                ),
            ],
            dim=-1,
        )

        distance_mat = torch.norm(
            template_translations.unsqueeze(-2) - template_translations.unsqueeze(-3), dim=-1
        )
        distance_mask = distance_mat <= self.distogram_max
        distogram_index = (
            torch.argmin(torch.abs(distance_mat.unsqueeze(-1) - self.distogram_bins), dim=-1)
            * distance_mask
            + len(self.distogram_bins) * ~distance_mask
        )
        template_distogram = F.one_hot(distogram_index, num_classes=len(self.distogram_bins) + 1)

        template_mask_pair = template_mask.unsqueeze(-1) & template_mask.unsqueeze(-2)
        return {
            "template_pair_feat": torch.cat(
                [template_distogram, template_aatype_pair, template_mask_pair.unsqueeze(-1)], dim=-1
            ).float(),
        }

    def parse_features(self, row):
        features = {
            "rotations": torch.from_numpy(
                np.stack(np.stack(row.rotations.tolist()).tolist())
            ).float(),
            "translations": torch.from_numpy(np.stack(row.translations.tolist())).float(),
            "local_coords": torch.tensor(
                [[p[1] for p in get_atom_positions(r)] for r in row.sequence]
            ),
            "target_feat": encode_one_hot(row.sequence),
            "positions": torch.tensor(row.positions),
            **self.parse_msa_features(row),
            **self.parse_template_features(row),
        }
        return features

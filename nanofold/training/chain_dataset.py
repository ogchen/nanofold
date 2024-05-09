import logging
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from nanofold.common.msa_metadata import COMPRESSED_MSA_FIELDS
from nanofold.common.residue_definitions import get_atom_positions
from nanofold.common.residue_definitions import RESIDUE_INDEX
from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA
from nanofold.training.frame import Frame


def accept_chain(length):
    return np.random.rand() < max(min(length, 512), 256) / 512


def quaternion_to_rotation_matrix(quaternion):
    quaternion = torch.concat([torch.ones(quaternion.size(0), 1), quaternion], dim=-1)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)

    a, b, c, d = (quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3])

    r0 = torch.stack([a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (a * c + b * d)], dim=-1)
    r1 = torch.stack([2 * (b * c + a * d), a**2 - b**2 + c**2 - d**2, 2 * (c * d - a * b)], dim=-1)
    r2 = torch.stack([2 * (b * d - a * c), 2 * (a * b + c * d), a**2 - b**2 - c**2 + d**2], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


def encode_one_hot(seq):
    indices = torch.tensor([RESIDUE_INDEX[residue] for residue in seq])
    return F.one_hot(indices, num_classes=len(RESIDUE_INDEX)).float()


class ChainDataset(IterableDataset):
    def __init__(self, table, indices, residue_crop_size, num_msa):
        super().__init__()
        self.residue_crop_size = residue_crop_size
        self.num_msa = num_msa
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
            "length",
            pc.list_value_length(table["positions"]),
        )
        return cls(table, indices[:train_size], *args, **kwargs), cls(
            table, indices[train_size:], *args, **kwargs
        )

    def extract_and_slice_msa(self, col_name, start, index, sequence_slice, length):
        sparse_matrix = torch.sparse_coo_tensor(
            self.table.column(f"{col_name}_coords")[index].as_py(),
            self.table.column(f"{col_name}_data")[index].as_py(),
            self.table.column(f"{col_name}_shape")[index].as_py(),
        )
        dense_matrix = (
            torch.stack([sparse_matrix[i] for i in range(start, start + length)])
            .to_dense()
            .reshape(length, -1, COMPRESSED_MSA_FIELDS[col_name].feat_size)
            .transpose(0, 1)
        )
        if sequence_slice is not None:
            dense_matrix = dense_matrix[:sequence_slice]
        return dense_matrix

    def __iter__(self):
        while True:
            sampled_index = np.random.choice(self.indices)
            length = self.table.column("length")[sampled_index].as_py()
            if not accept_chain(length):
                continue

            start = np.random.randint(max(1, length - self.residue_crop_size + 1))
            length = min(length, self.residue_crop_size)

            slice_column_list = lambda col_name: pa.compute.list_slice(
                self.table.column(col_name)[sampled_index], start, start + length
            ).as_py()
            slice_column_str = lambda col_name: pa.compute.utf8_slice_codeunits(
                self.table.column(col_name)[sampled_index], start, start + length
            ).as_py()
            slice_column_nested_str = lambda col_name: [
                pa.compute.utf8_slice_codeunits(x, start, start + length).as_py()
                for x in self.table.column(col_name)[sampled_index]
            ]
            slice_column_nested_list = lambda col_name, max_sequences=None: [
                pa.compute.list_slice(x, start, start + length).as_py()
                for x in (
                    pa.compute.list_slice(
                        self.table.column(col_name)[sampled_index], 0, max_sequences
                    )
                    if max_sequences is not None
                    else self.table.column(col_name)[sampled_index]
                )
            ]

            row = {
                "positions": slice_column_list("positions"),
                "translations": slice_column_list("translations"),
                "rotations": slice_column_list("rotations"),
                "sequence": slice_column_str("sequence"),
                "template_mask": slice_column_nested_list("template_mask"),
                "template_sequence": slice_column_nested_str("template_sequence"),
                "template_translations": slice_column_nested_list("template_translations"),
                "template_rotations": slice_column_nested_list("template_rotations"),
                "profile": slice_column_list("profile"),
                "deletion_mean": slice_column_list("deletion_mean"),
            } | {
                msa_field: self.extract_and_slice_msa(
                    msa_field, start, sampled_index, self.num_msa, length
                )
                for msa_field in COMPRESSED_MSA_FIELDS.keys()
            }
            yield self.parse_features(row, length)

    def parse_template_features(self, row, length):
        if len(row["template_sequence"]) == 0:
            return {
                "template_pair_feat": torch.empty((0, length, length, 84)),
            }
        template_backbone_frame_mask = torch.tensor(row["template_mask"])
        template_translations = torch.tensor(row["template_translations"])
        template_rotations = torch.tensor(row["template_rotations"])
        frames = Frame(template_rotations.unsqueeze(-3), template_translations.unsqueeze(-2))
        template_unit_vector = Frame.apply(
            Frame.inverse(frames), template_translations.unsqueeze(-3)
        )

        aatype_index = torch.tensor(
            [[RESIDUE_INDEX_MSA[residue] for residue in seq] for seq in row["template_sequence"]]
        )
        template_restype = F.one_hot(aatype_index, num_classes=len(RESIDUE_INDEX_MSA))

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

        return {
            "template_restype": template_restype,
            "template_backbone_frame_mask": template_backbone_frame_mask,
            "template_distogram": template_distogram,
            "template_unit_vector": template_unit_vector,
        }

    def parse_features(self, row, length):
        local_coords = torch.tensor(
            [[p[1] for p in get_atom_positions(r)] for r in row["sequence"]]
        )
        residue_index = torch.tensor(row["positions"])
        random_quaternions = torch.rand(local_coords.size(0), 3) * 100
        random_rotations = quaternion_to_rotation_matrix(random_quaternions)
        random_translations = torch.rand(local_coords.size(0), 3) * 100
        frames = Frame(random_rotations.unsqueeze(-3), random_translations.unsqueeze(-2))
        ref_pos = Frame.apply(frames, local_coords).view(-1, 3)
        ref_space_uid = (
            residue_index.unsqueeze(-1)
            .expand(*residue_index.size(), local_coords.size(-1))
            .reshape(-1)
        )

        features = {
            "rotations": torch.tensor(row["rotations"]),
            "translations": torch.tensor(row["translations"]),
            "local_coords": local_coords,
            "residue_index": residue_index,
            "restype": encode_one_hot(row["sequence"]),
            "ref_pos": ref_pos,
            "ref_space_uid": ref_space_uid,
            "msa": row["msa"],
            "has_deletion": row["has_deletion"],
            "deletion_value": row["deletion_value"],
            "profile": torch.tensor(row["profile"]),
            "deletion_mean": torch.tensor(row["deletion_mean"]),
            **self.parse_template_features(row, length),
        }
        return features

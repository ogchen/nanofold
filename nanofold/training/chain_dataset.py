from torch.utils.data import IterableDataset
import logging
import numpy as np
import polars as pl
import torch

from nanofold.common.residue_definitions import BACKBONE_POSITIONS
from nanofold.common.residue_definitions import RESIDUE_LOOKUP_1L
from nanofold.training.model.input import encode_one_hot


SAMPLE_SIZE = 100


class ChainDataset(IterableDataset):
    def __init__(self, df, residue_crop_size):
        super().__init__()
        self.residue_crop_size = residue_crop_size
        self.df = df
        self.df = self.df.with_columns(length=pl.col("sequence").str.len_chars())
        self.df = self.df.filter(pl.col("length") >= self.residue_crop_size)

    @classmethod
    def construct_datasets(cls, arrow_file, train_split, residue_crop_size):
        df = pl.read_ipc(arrow_file, columns=["rotations", "translations", "sequence", "positions"])
        logging.info(f"Dataframe loaded, estimated size {df.estimated_size(unit="mb"):.2f} MB")
        train_size = int(train_split * len(df))
        if train_size <= 0 or train_split >= len(df):
            raise ValueError(f"train_size must be between 0 and len(df), got {train_size}")
        df = df.sample(fraction=1)
        return cls(df.head(train_size), residue_crop_size), cls(
            df.tail(-train_size), residue_crop_size
        )

    def __iter__(self):
        while True:
            sample = self.df.sample(SAMPLE_SIZE, with_replacement=True, shuffle=True)
            sample = sample.with_columns(
                start=pl.lit(np.random.randint(sample["length"] - self.residue_crop_size + 1)),
            )
            sample = sample.with_columns(
                positions=pl.col("positions").list.slice(pl.col("start"), self.residue_crop_size),
                sequence=pl.col("sequence").str.slice(pl.col("start"), self.residue_crop_size),
                translations=pl.col("translations").list.slice(
                    pl.col("start") * 3, self.residue_crop_size * 3
                ),
                rotations=pl.col("rotations").list.slice(
                    pl.col("start") * 3 * 3, self.residue_crop_size * 3 * 3
                ),
            )
            for row in sample.iter_rows(named=True):
                yield ChainDataset.process_row(row)

    @staticmethod
    def process_row(row):
        row["rotations"] = torch.tensor(row["rotations"]).reshape(-1, 3, 3)
        row["translations"] = torch.tensor(row["translations"]).reshape(-1, 3)
        row["positions"] = torch.tensor(row["positions"])
        row["target_feat"] = encode_one_hot(row["sequence"])
        row["local_coords"] = torch.tensor(
            [[a[1] for a in BACKBONE_POSITIONS[RESIDUE_LOOKUP_1L[r]]] for r in row["sequence"]]
        )
        return row

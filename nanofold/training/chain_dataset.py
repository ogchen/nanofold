from torch.utils.data import IterableDataset
import numpy as np
import polars as pl
import torch


class ChainDataset(IterableDataset):
    def __init__(self, arrow_file, residue_crop_size, batch_size):
        super().__init__()
        self.residue_crop_size = residue_crop_size
        self.batch_size = batch_size
        self.df = pl.read_ipc(
            arrow_file, columns=["rotations", "translations", "sequence", "positions"]
        )
        self.df = self.df.with_columns(length=pl.col("sequence").str.len_chars())
        self.df = self.df.filter(pl.col("length") >= self.residue_crop_size)

    def __iter__(self):
        while True:
            sample = self.df.sample(n=self.batch_size, shuffle=True)
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
            batch = {
                "rotations": torch.stack(
                    [torch.tensor(r.to_numpy().reshape(-1, 3, 3)) for r in sample["rotations"]]
                ),
                "translations": torch.stack(
                    [torch.tensor(t.to_numpy().reshape(-1, 3)) for t in sample["translations"]]
                ),
                "sequence": sample["sequence"].to_list(),
                "positions": torch.stack([torch.tensor(p.to_numpy()) for p in sample["positions"]]),
            }
            yield batch

from torch.utils.data import Dataset
import pyarrow as pa

from nanofold.common.chain_record import ChainRecord


class ChainDataset(Dataset):
    def __init__(self, arrow_file):
        self.mmap = pa.memory_map(str(arrow_file), mode="r")
        self.reader = pa.ipc.open_file(self.mmap)
        self.table = self.reader.read_all()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError(f"Expected int, got {type(idx)}")
        batches = self.table.take([idx]).to_batches()
        if len(batches) != 1:
            raise ValueError(f"Expected 1 batch when reading index {idx}, got {len(batches)}")
        chains = ChainRecord.from_record_batch(batches[0])
        if len(chains) != 1:
            raise ValueError(f"Expected 1 chain when reading index {idx}, got {len(chains)}")
        return chains[0]

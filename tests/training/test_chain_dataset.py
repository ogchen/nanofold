import pytest
import torch

from nanofold.training.chain_dataset import ChainDataset


@pytest.fixture
def arrow_file(request):
    return request.path.parent / "data" / "pdb_data.test.arrow"


def test_chain_dataset(arrow_file):
    residue_crop_size = 32
    batch_size = 2
    dataset = iter(ChainDataset(arrow_file, residue_crop_size, batch_size))
    batch = next(dataset)
    assert batch["rotations"].shape == (batch_size, residue_crop_size, 3, 3)
    assert batch["translations"].shape == (batch_size, residue_crop_size, 3)
    assert len(batch["sequence"]) == batch_size
    assert len(batch["sequence"][0]) == residue_crop_size
    assert batch["positions"].shape == (batch_size, residue_crop_size)
    assert torch.allclose(
        batch["rotations"] @ batch["rotations"].transpose(-1, -2), torch.eye(3), atol=1e-3
    )

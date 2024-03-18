import pytest
import torch

from nanofold.training.chain_dataset import ChainDataset


@pytest.fixture
def arrow_file(request):
    return request.path.parent / "data" / "pdb_data.test.arrow"


def test_chain_dataset(arrow_file):
    train_split = 0.8
    residue_crop_size = 32
    train_data, test_data = ChainDataset.construct_datasets(
        arrow_file, train_split, residue_crop_size, device="cpu"
    )
    assert len(train_data.df) == 8
    assert len(test_data.df) == 2

    batch = next(iter(train_data))
    assert batch["rotations"].shape == (residue_crop_size, 3, 3)
    assert batch["translations"].shape == (residue_crop_size, 3)
    assert len(batch["sequence"]) == residue_crop_size
    assert batch["positions"].shape == (residue_crop_size,)
    assert batch["target_feat"].shape == (residue_crop_size, 20)
    assert torch.allclose(
        batch["rotations"] @ batch["rotations"].transpose(-1, -2), torch.eye(3), atol=1e-3
    )

import pytest
import torch

from nanofold.training.chain_dataset import encode_one_hot
from nanofold.training.chain_dataset import ChainDataset
from nanofold.common.residue_definitions import RESIDUE_INDEX


@pytest.fixture
def arrow_file(request):
    return request.path.parent / "data" / "features.test.arrow"


def test_chain_dataset(arrow_file):
    train_split = 0.8
    residue_crop_size = 32
    num_msa = 16
    train_data, test_data = ChainDataset.construct_datasets(
        arrow_file, train_split, residue_crop_size, num_msa
    )
    assert len(train_data.df) == 10
    assert len(test_data.df) == 10

    assert len(train_data.indices) == 8
    assert len(test_data.indices) == 2

    batch = next(iter(train_data))
    assert batch["rotations"].shape == (residue_crop_size, 3, 3)
    assert batch["translations"].shape == (residue_crop_size, 3)
    assert batch["positions"].shape == (residue_crop_size,)
    assert batch["target_feat"].shape == (residue_crop_size, len(RESIDUE_INDEX))
    assert batch["msa_feat"].shape[:2] == (num_msa, residue_crop_size)
    assert torch.allclose(
        batch["rotations"] @ batch["rotations"].transpose(-1, -2), torch.eye(3), atol=1e-3
    )


def test_encode_one_hot():
    seq = "ADHIAA"
    one_hot = encode_one_hot(seq)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert torch.equal(one_hot, expected)

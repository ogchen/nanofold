import pytest
from nanofold.training.chain_dataset import ChainDataset


@pytest.fixture
def arrow_file(request):
    return request.path.parent / "data" / "pdb_data.test.arrow"


def test_chain_dataset(arrow_file):
    dataset = ChainDataset(arrow_file)
    assert len(dataset) == 13
    chain = dataset[10]
    assert chain.model_id == "1RDT"
    assert len(chain.positions) == 7
    assert chain.rotations.shape == (7, 3, 3)
    assert chain.translations.shape == (7, 3)
    with pytest.raises(IndexError):
        dataset[len(dataset)]

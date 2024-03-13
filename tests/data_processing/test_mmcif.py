import pytest
from nanofold.data_processing import mmcif


@pytest.mark.parametrize(
    "model, num_chains, num_residues",
    [("1A00", 4, 141), ("1YUJ", 3, 0)],
    indirect=["model"],
)
def test_get_residue(model, num_chains, num_residues):
    chains = list(model.get_chains())
    assert len(chains) == num_chains
    metadata, rotations, translations = mmcif.get_residues(chains[0])
    assert len(metadata) == num_residues
    assert len(rotations) == num_residues
    assert len(translations) == num_residues

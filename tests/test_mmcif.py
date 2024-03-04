import torch
from nanofold import mmcif


def test_load_model(model):
    chains = list(model.get_chains())
    assert len(chains) == 4
    residues = list(mmcif.get_residues(chains[0]))
    for r in residues:
        result = r["rotation"] @ r["rotation"].transpose(-2, -1)
        assert torch.allclose(result, torch.eye(3), atol=1e-5)
    assert len(residues) == 141


def test_parse_chains(model):
    chains = mmcif.parse_chains(model)
    assert len(chains) == 2
    assert chains[0].id == ("1A00", "A")
    assert (
        chains[0].sequence
        == "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    )
    assert len(chains[0].chain) == 141
    assert chains[0].frames.rotations.shape == (141, 3, 3)


def test_get_residue(model):
    chains = list(model.get_chains())
    assert len(chains) == 4
    residues = list(mmcif.get_residues(chains[0]))
    for r in residues:
        result = r["rotation"] @ r["rotation"].transpose(-2, -1)
        assert torch.allclose(result, torch.eye(3), atol=1e-5)
    assert len(residues) == 141

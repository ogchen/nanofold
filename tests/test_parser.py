import pytest
import torch
from pathlib import Path
from nanofold import parser


@pytest.fixture
def data_dir(request):
    return request.path.parent / "data"


@pytest.fixture
def model(data_dir):
    identifiers = parser.list_available_mmcif(data_dir)
    assert len(identifiers) == 1
    assert identifiers[0]["id"] == "1A00"
    return parser.load_model(identifiers[0]["id"], identifiers[0]["filepath"])


def test_load_model(model):
    assert model.header["idcode"] == "1A00"
    chains = list(model.get_chains())
    assert len(chains) == 4
    residues = list(parser.get_residues(chains[0]))
    for r in residues:
        result = r["rotation"] @ r["rotation"].transpose(-2, -1)
        assert torch.allclose(result, torch.eye(3), atol=1e-5)
    assert len(residues) == 141


def test_get_fasta(data_dir, model):
    fasta_file = data_dir / "pdb_seqres.txt"
    fasta_parser = parser.FastaParser(fasta_file)
    chain_id = next(model.get_chains()).get_full_id()
    fasta = fasta_parser.get_fasta(chain_id[0], chain_id[2])
    assert len(fasta) == 141


def test_get_frames(model):
    chain = next(model.get_chains())
    frames = parser.get_frames(chain)
    assert frames.translations.shape == (141, 3)
    assert frames.rotations.shape == (141, 3, 3)

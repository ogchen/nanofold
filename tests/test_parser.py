import pytest
from pathlib import Path
from nanofold import parser


@pytest.fixture
def data_dir(request):
    return request.path.parent / "data"


@pytest.fixture
def mmcif_structure(data_dir):
    identifiers = parser.list_available_mmcif(data_dir)
    assert len(identifiers) == 1
    assert identifiers[0]["id"] == "1A00"
    return parser.load_structure(identifiers[0]["id"], identifiers[0]["filepath"])


def test_load_structure(mmcif_structure):
    assert mmcif_structure.header["idcode"] == "1A00"
    chains = list(mmcif_structure.get_chains())
    assert len(chains) == 4
    c_alphas = list(parser.get_c_alphas(chains[0]))
    assert len(c_alphas) == 141


def test_get_fasta(data_dir, mmcif_structure):
    fasta_file = data_dir / "pdb_seqres.txt"
    fasta_parser = parser.FastaParser(fasta_file)
    chain_id = next(mmcif_structure.get_chains()).get_full_id()
    fasta = fasta_parser.get_fasta(chain_id[0], chain_id[2])
    assert len(fasta) == 141
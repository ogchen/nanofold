import pytest
from nanofold.training.fasta import FastaParser


@pytest.fixture
def data_dir(request):
    return request.path.parent / "data"


def test_get_fasta(data_dir):
    fasta_file = data_dir / "pdb_seqres.txt"
    fasta_parser = FastaParser(fasta_file)
    seq = fasta_parser.get_fasta("1A00", "A")
    assert len(seq) == 141

from nanofold.fasta import FastaParser


def test_get_fasta(data_dir):
    fasta_file = data_dir / "pdb_seqres.txt"
    fasta_parser = FastaParser(fasta_file)
    seq = fasta_parser.get_fasta("1A00", "A")
    assert len(seq) == 141

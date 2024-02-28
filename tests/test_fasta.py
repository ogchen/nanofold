from nanofold.fasta import FastaParser


def test_get_fasta(data_dir, model):
    fasta_file = data_dir / "pdb_seqres.txt"
    fasta_parser = FastaParser(fasta_file)
    chain_id = next(model.get_chains()).get_full_id()
    seq = fasta_parser.get_fasta(chain_id[0], chain_id[2])
    assert len(seq) == 141

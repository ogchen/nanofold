from Bio import SeqIO


class FastaParser:
    def __init__(self, fasta_file):
        self.fasta_file = fasta_file

    def get_fasta(self, structure_id, chain_id):
        fasta_id = f"{structure_id.lower()}_{chain_id.upper()}"
        with open(self.fasta_file, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                if record.id == fasta_id:
                    return record
        raise RuntimeError(f"Could not find fasta sequence for {fasta_id}")

import glob
import os
from Bio import SeqIO
from Bio.PDB import MMCIFParser
from pathlib import Path


def list_available_mmcif(mmcif_dir):
    search_glob = os.path.join(mmcif_dir, "*.cif")
    mmcif_files = glob.glob(search_glob)
    identifiers = [{"id": Path(f).stem, "filepath": f} for f in mmcif_files]
    return identifiers


def load_structure(id, filepath):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(id, filepath)
    return structure


def get_c_alphas(chain):
    for residue in chain.get_residues():
        for atom in residue.get_atoms():
            if atom.get_name() == "CA":
                yield {
                    "resname": residue.get_resname(),
                    "id": residue.get_id(),
                    "coord": atom.get_coord(),
                }


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

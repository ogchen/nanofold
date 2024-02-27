import glob
import numpy as np
import os
import torch
from Bio import SeqIO
from Bio.PDB import MMCIFParser
from pathlib import Path
from nanofold.frame import Frame


def list_available_mmcif(mmcif_dir):
    search_glob = os.path.join(mmcif_dir, "*.cif")
    mmcif_files = glob.glob(search_glob)
    identifiers = [{"id": Path(f).stem, "filepath": f} for f in mmcif_files]
    return identifiers


def load_model(id, filepath):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(id, filepath)
    try:
        model = next(structure.get_models())
        model.header = structure.header
    except StopIteration:
        raise RuntimeError(f"No models found in {filepath}")
    return model


def get_c_alphas(chain):
    for residue in chain.get_residues():
        for atom in residue.get_atoms():
            if atom.get_name() == "CA":
                yield {
                    "resname": residue.get_resname(),
                    "id": residue.get_id(),
                    "coord": atom.get_coord(),
                }


def get_c_alpha_coords(chain):
    return torch.from_numpy(
        np.stack([record["coord"] for record in get_c_alphas(chain)])
    )


def get_frames(chain):
    translations = get_c_alpha_coords(chain)
    rotations = torch.stack([torch.eye(3) for _ in range(len(translations))])
    return Frame(rotations, translations)


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

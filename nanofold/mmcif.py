import glob
import os
import torch
from Bio.PDB import MMCIFParser
from pathlib import Path
from nanofold.frame import Frame
from nanofold.residue import RESIDUE_LIST
from nanofold.residue import compute_residue_rotation


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


def get_residues(chain):
    valid_residues = [res[1] for res in RESIDUE_LIST]
    result = []
    for residue in chain.get_residues():
        if residue.get_resname() not in valid_residues:
            continue
        ca = None
        c = None
        n = None
        for atom in residue.get_atoms():
            ca = atom if atom.get_name() == "CA" else ca
            c = atom if atom.get_name() == "C" else c
            n = atom if atom.get_name() == "N" else n
            if all((x is not None for x in [ca, c, n])):
                break
        ca_coords = torch.from_numpy(ca.get_coord())
        result.append(
            {
                "resname": residue.get_resname(),
                "id": residue.get_id(),
                "rotation": compute_residue_rotation(
                    n_coords=torch.from_numpy(n.get_coord()),
                    ca_coords=ca_coords,
                    c_coords=torch.from_numpy(c.get_coord()),
                ),
                "translation": ca_coords,
            }
        )
    return result


def get_frames(residues):
    translations = torch.stack([r["translation"] for r in residues])
    rotations = torch.stack([r["rotation"] for r in residues])
    return Frame(rotations, translations)

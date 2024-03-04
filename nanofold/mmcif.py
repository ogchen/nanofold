import glob
import os
import torch
from Bio.PDB import MMCIFParser
from nanofold.chain import Chain
from nanofold.residue import compute_residue_rotation
from nanofold.residue import RESIDUE_LIST


def list_available_mmcif(mmcif_dir):
    search_glob = os.path.join(mmcif_dir, "*.cif")
    return glob.glob(search_glob)


def load_model(filepath):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(id, filepath)
    try:
        model = next(structure.get_models())
    except StopIteration:
        raise RuntimeError(f"No models found in {filepath}")
    model.id = parser._mmcif_dict["_entry.id"][0]
    model.header = structure.header
    model.mmcif_dict = parser._mmcif_dict
    return model


def parse_chains(model):
    result = []
    for strand_id, sequence in zip(
        model.mmcif_dict["_entity_poly.pdbx_strand_id"],
        model.mmcif_dict["_entity_poly.pdbx_seq_one_letter_code"],
    ):
        strand_id = strand_id.split(",")[0]
        sequence = sequence.replace("\n", "")
        mmcif_chain = model[strand_id]
        residue_list = get_residues(mmcif_chain)
        if len(residue_list) == 0:
            continue
        _, structure_id, chain_id = mmcif_chain.get_full_id()
        result.append(Chain((structure_id, chain_id), sequence, residue_list))
    if len(result) == 0:
        raise RuntimeError(f"No valid chains found for model {model.id}")
    return result


def should_filter_residue(residue):
    valid_residues = [r[1] for r in RESIDUE_LIST]
    hetatom, _, _ = residue.get_id()
    is_hetero_residue = hetatom.strip() != ""
    return is_hetero_residue or residue.get_resname() not in valid_residues


def get_residues(chain):
    result = []
    for residue in chain.get_residues():
        if should_filter_residue(residue):
            continue
        if "N" not in residue or "CA" not in residue or "C" not in residue:
            raise RuntimeError(
                f"Missing backbone atoms for residue {residue.get_full_id()}"
            )
        n_coords = torch.from_numpy(residue["N"].get_coord())
        ca_coords = torch.from_numpy(residue["CA"].get_coord())
        c_coords = torch.from_numpy(residue["C"].get_coord())
        result.append(
            {
                "resname": residue.get_resname(),
                "id": residue.get_full_id(),
                "rotation": compute_residue_rotation(
                    n_coords=n_coords,
                    ca_coords=ca_coords,
                    c_coords=c_coords,
                ),
                "translation": ca_coords,
            }
        )
    return result

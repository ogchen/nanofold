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


def check_chain(residue_list, chain, sequence):
    get_resseq = lambda i: residue_list[i]["id"][-1][1]
    expected_length = get_resseq(-1) - get_resseq(0) + 1
    if len(chain.sequence) != expected_length:
        raise RuntimeError(
            f"Sequence length mismatch for chain {chain.id} (expected {expected_length}, got {len(chain.sequence)})"
        )
    if chain.sequence not in sequence:
        raise RuntimeError(f"Sequence mismatch for chain {chain.id}")


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
        chain = Chain(mmcif_chain.get_full_id()[1:], residue_list)

        # Sanity checks on sequence and length
        check_chain(residue_list, chain, sequence)
        result.append(chain)

    if len(result) == 0:
        raise RuntimeError(f"No valid chains found for model {model.id}")
    return result


def should_filter_residue(residue):
    valid_residues = [r[1] for r in RESIDUE_LIST]
    hetatom, _, _ = residue.get_id()
    is_hetero_residue = hetatom.strip() != ""
    is_valid_residue = residue.get_resname() in valid_residues
    return is_hetero_residue or not is_valid_residue


def get_residues(chain):
    result = []
    for residue in chain.get_residues():
        if should_filter_residue(residue):
            continue
        if "N" not in residue or "CA" not in residue or "C" not in residue:
            raise RuntimeError(
                f"Missing backbone atoms for residue {residue.get_full_id()[1:]}"
            )
        n_coords = torch.from_numpy(residue["N"].get_coord())
        ca_coords = torch.from_numpy(residue["CA"].get_coord())
        c_coords = torch.from_numpy(residue["C"].get_coord())
        result.append(
            {
                "resname": residue.get_resname(),
                "id": residue.get_full_id()[1:],
                "rotation": compute_residue_rotation(
                    n_coords=n_coords,
                    ca_coords=ca_coords,
                    c_coords=c_coords,
                ),
                "translation": ca_coords,
            }
        )
    return result

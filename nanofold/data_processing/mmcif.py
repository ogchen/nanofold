import difflib
import glob
import os
import numpy as np
from Bio.PDB import MMCIFParser

from nanofold.common.residue import RESIDUE_LOOKUP_3L
from nanofold.data_processing.chain import Chain
from nanofold.data_processing.residue import compute_residue_frames


class EmptyChainError(RuntimeError):
    pass


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


def get_longest_match(chain, sequence):
    matcher = difflib.SequenceMatcher(None, chain.sequence, sequence, autojunk=False)
    start, _, size = matcher.find_longest_match()
    return chain[start : start + size]


def parse_chains(model):
    result = []
    for strand_id, sequence in zip(
        model.mmcif_dict["_entity_poly.pdbx_strand_id"],
        model.mmcif_dict["_entity_poly.pdbx_seq_one_letter_code"],
    ):
        strand_id = strand_id.split(",")[0]
        sequence = sequence.replace("\n", "")
        mmcif_chain = model[strand_id]
        chain = Chain.from_residue_list(mmcif_chain.get_full_id()[1:], *get_residues(mmcif_chain))
        chain = get_longest_match(chain, sequence)
        if len(chain) == 0:
            continue
        result.append(chain)

    if len(result) == 0:
        raise EmptyChainError(f"No valid chains found for model {model.id}")
    return result


def should_filter_residue(residue):
    hetatom, _, _ = residue.get_id()
    is_hetero_residue = hetatom.strip() != ""
    is_valid_residue = residue.get_resname() in RESIDUE_LOOKUP_3L.keys()
    return is_hetero_residue or not is_valid_residue


def get_residues(chain):
    atoms = ["N", "CA", "C"]
    metadata = []
    coords = np.empty((0, 3, 3))
    for residue in chain.get_residues():
        if should_filter_residue(residue):
            continue
        if any(a not in residue for a in atoms):
            continue
        residue_coords = np.stack([residue[a].get_coord() for a in atoms])
        coords = np.concatenate([coords, residue_coords[np.newaxis, :]])
        metadata.append(
            {
                "resname": residue.get_resname(),
                "id": residue.get_full_id()[1:],
                "serial_number": residue["CA"].get_serial_number(),
            }
        )
    return metadata, *compute_residue_frames(coords)

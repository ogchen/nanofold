import logging
import numpy as np
from Bio.PDB import MMCIFParser
from pathlib import Path

from nanofold.common.residue_definitions import get_1l_res_code
from nanofold.common.residue_definitions import BACKBONE_ATOMS
from nanofold.data_processing.residue import compute_residue_frames


def get_model_id(filepath):
    return Path(filepath).stem.lower()


def parse_mmcif_file(filepath, capture_errors):
    try:
        parser = MMCIFParser(QUIET=True)
        label_parser = MMCIFParser(QUIET=True, auth_residues=False)
        structure = parser.get_structure(get_model_id(filepath), filepath)
        label_structure = label_parser.get_structure(get_model_id(filepath), filepath)
        return parse_structure(structure, label_structure)
    except Exception as e:
        logging.warning(f"Caught exception for file={filepath}, error={e}")
        if not capture_errors:
            raise e
        return []


def parse_structure(structure, label_structure):
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Multi model mmCIF files are not supported, got {len(models)}")
    result = [
        parse_chain(chain, label_chain) for chain, label_chain in zip(models[0], label_structure[0])
    ]
    return [c for c in result if len(c["sequence"]) > 0]


def parse_chain(chain, label_chain):
    structure_id, _, chain_id = chain.get_full_id()
    positions = []
    label_positions = []
    res_codes = []
    coords = np.empty((0, 3, 3))
    for residue, label_residue in zip(chain, label_chain):
        _, position, insert_code = residue.get_id()
        if insert_code != " ":
            raise ValueError(f"Insert codes are not supported: {insert_code}")
        if any(a not in residue for a in BACKBONE_ATOMS):
            continue
        residue_coords = np.stack([residue[a].get_coord() for a in BACKBONE_ATOMS])
        coords = np.concatenate([coords, residue_coords[np.newaxis, :]])
        positions.append(position)
        res_codes.append(get_1l_res_code(residue.get_resname()))
        label_positions.append(label_residue.get_id()[1])
    rotations, translations = compute_residue_frames(coords)
    return {
        "structure_id": structure_id,
        "chain_id": chain_id,
        "sequence": "".join(res_codes),
        "positions": positions,
        "label_positions": label_positions,
        "rotations": rotations,
        "translations": translations,
    }

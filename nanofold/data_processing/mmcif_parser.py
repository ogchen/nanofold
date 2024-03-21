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
        structure = parser.get_structure(get_model_id(filepath), filepath)
        return parse_structure(structure)
    except Exception as e:
        logging.warning(f"Caught exception for file={filepath}, error={e}")
        if not capture_errors:
            raise e
        return []


def parse_structure(structure):
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Expected 1 model, got {len(models)} for {structure.header['idcode']}")
    result = [parse_chain(chain) for chain in models[0]]
    return [c for c in result if len(c["sequence"]) > 0]


def parse_chain(chain):
    structure_id, _, chain_id = chain.get_full_id()
    positions = []
    res_codes = []
    coords = np.empty((0, 3, 3))
    for residue in chain:
        _, position, insert_code = residue.get_id()
        if insert_code != " ":
            raise ValueError(f"Insert codes are not supported: {insert_code}")
        if any(a not in residue for a in BACKBONE_ATOMS):
            continue
        residue_coords = np.stack([residue[a].get_coord() for a in BACKBONE_ATOMS])
        coords = np.concatenate([coords, residue_coords[np.newaxis, :]])
        positions.append(position)
        res_codes.append(get_1l_res_code(residue.get_resname()))
    rotations, translations = compute_residue_frames(coords)
    return {
        "structure_id": structure_id,
        "chain_id": chain_id,
        "sequence": "".join(res_codes),
        "positions": positions,
        "rotations": rotations,
        "translations": translations,
    }

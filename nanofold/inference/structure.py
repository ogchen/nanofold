from Bio.PDB.StructureBuilder import StructureBuilder

from nanofold.common.residue_definitions import get_3l_res_name
from nanofold.common.residue_definitions import BACKBONE_POSITIONS


def coords_to_bio_structure(sequence, coords):
    structure_builder = StructureBuilder()
    structure_builder.init_structure("PHA-L")
    structure_builder.init_model(0)
    structure_builder.init_chain("A")
    structure_builder.init_seg(" ")
    for i, r in enumerate(sequence):
        res_name = get_3l_res_name(r)
        structure_builder.init_residue(res_name, " ", i + 1, " ")
        for meta, c in zip(BACKBONE_POSITIONS[res_name], coords[i]):
            structure_builder.init_atom(meta[0], c, 1.0, 1.0, " ", meta[0])
    return structure_builder.get_structure()

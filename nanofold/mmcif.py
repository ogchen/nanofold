import glob
import os
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

import string
from collections import OrderedDict


def compute_deletion_matrix(alignments):
    deletion_matrix = []
    for seq in alignments:
        deletion_row = []
        deletion_count = 0
        for c in seq:
            if c.islower():
                deletion_count += 1
            else:
                deletion_row.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_row)
    return deletion_matrix


def extract_alignments(input):
    alignments = OrderedDict()
    current_id = None
    for line in input:
        line = line.strip()
        if line.startswith(">"):
            current_id = line[1:]
        elif current_id is not None and line:
            alignments[current_id] = line
    alignments = list(alignments.values())
    deletion_matrix = compute_deletion_matrix(alignments)
    table = str.maketrans("", "", string.ascii_lowercase)
    alignments = [a.translate(table) for a in alignments]
    for a in alignments:
        if len(a) != len(alignments[0]):
            raise ValueError("MSA sequences are not of equal length")

    return alignments, deletion_matrix

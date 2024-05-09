import numpy as np
import subprocess
from collections import OrderedDict
from tempfile import NamedTemporaryFile


def is_alignment_line(line):
    return not line.startswith(("#", "//")) and not line.isspace()


def extract_sequence_names(input, max_sequences):
    input.seek(0)
    sequences = set()
    for line in input:
        if is_alignment_line(line):
            sequences.add(line.split()[0])
            if len(sequences) >= max_sequences:
                break
    return sequences


def compress_alignment_gaps(alignments):
    query_mask = [i for i, c in enumerate(alignments[0]) if c != "-"]
    return ["".join([a[i] for i in query_mask]) for a in alignments]


def compute_deletion_matrix(alignments):
    query = alignments[0]
    deletion_matrix = []

    for sequence in alignments:
        deletion_row = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query):
            if seq_res != "-" and query_res == "-":
                deletion_count += 1
            elif query_res != "-":
                deletion_row.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_row)
    return deletion_matrix


def extract_alignments(input):
    alignments = OrderedDict()
    for line in input:
        if is_alignment_line(line):
            line = line.split()
            alignments[line[0]] = alignments.get(line[0], "") + line[1]
    alignments = list(alignments.values())
    if not all([len(a) == len(alignments[0]) for a in alignments]):
        raise ValueError("MSA sequences are not of equal length")
    deletion_matrix = compute_deletion_matrix(alignments)
    return compress_alignment_gaps(alignments), deletion_matrix


def filter_sto_by_sequences(input, sequences):
    input.seek(0)
    keep_line = lambda line: any(
        [
            line.isspace(),
            line.strip() == "//",
            line.startswith("# STOCKHOLM"),
            line.startswith("#=GC RF"),
            line.startswith("#=GS") and line.split()[1] in sequences,
            not (line.startswith("#") or line.isspace()) and line.split()[0] in sequences,
        ]
    )
    return "".join([line for line in input if keep_line(line)])


def truncate_sto(input, max_sequences):
    sequences = extract_sequence_names(input, max_sequences)
    return filter_sto_by_sequences(input, sequences)


def convert_to_a2m(reformat_bin, msa_sto, a2m_file):
    with NamedTemporaryFile(mode="w") as tmp:
        for s in msa_sto:
            tmp.write(s)
        tmp.flush()
        cmd = [reformat_bin, "sto", "a2m", tmp.name, a2m_file]
        subprocess.run(cmd, capture_output=True, check=True)

import numpy as np
from collections import OrderedDict


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


def extract_alignments(input):
    input.seek(0)
    alignments = OrderedDict()
    for line in input:
        if is_alignment_line(line):
            line = line.split()
            alignments[line[0]] = alignments.get(line[0], "") + line[1]
    return list(alignments.values())


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


def compress_alignment_gaps(alignments):
    query_mask = [i for i, c in enumerate(alignments[0]) if c != "-"]
    return ["".join([a[i] for i in query_mask]) for a in alignments]


def compute_deletion_matrix(alignments):
    query = alignments[0]
    deletion_matrix = []

    for sequence in alignments:
        deletion_vec = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query):
            if seq_res != "-" and query_res == "-":
                deletion_count += 1
            elif query_res != "-":
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)
    return deletion_matrix


def parse_msa(input, num_samples=None):
    alignments = extract_alignments(input)
    if not all([len(a) == len(alignments[0]) for a in alignments]):
        raise ValueError("MSA sequences are not of equal length")
    if num_samples is not None and len(alignments) > num_samples:
        alignments = [
            alignments[0],
            *np.random.choice(alignments, num_samples - 1, replace=False).tolist(),
        ]
    compressed_alignments = compress_alignment_gaps(alignments)
    deletion_matrix = compute_deletion_matrix(alignments)
    return {
        "alignments": compressed_alignments,
        "deletion_matrix": deletion_matrix,
    }

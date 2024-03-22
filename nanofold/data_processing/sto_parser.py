def extract_sequences(input, max_sequences):
    input.seek(0)
    sequences = set()
    for line in input:
        if line.isspace() or line.startswith(("#", "//")):
            continue
        sequences.add(line.split()[0])
        if len(sequences) >= max_sequences:
            break
    return sequences


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
    sequences = extract_sequences(input, max_sequences)
    return filter_sto_by_sequences(input, sequences)

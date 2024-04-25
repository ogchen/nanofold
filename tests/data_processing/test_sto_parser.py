from io import StringIO

from nanofold.data_processing.sto_parser import extract_alignments


def test_parse_msa():
    input = StringIO(
        """
# STOCKHOLM 1.0

#=GS M9YY75_9ADEN/375-612       DE [subseq from] M9YY75_9ADEN
#=GS A0A0K0PX17_9ADEN/146-396   DE [subseq from] A0A0K0PX17_9ADEN
#=GS Q8QVG3_ADEBA/907-1142      DE [subseq from] Q8QVG3_ADEBA

1qiu_C                             FDNTAIAINAGKGLEFD------TN---TS
M9YY75_9ADEN/375-612               ----QLRLNIGQGLRYN------P----TS
A0A0K0PX17_9ADEN/146-396           ------TKEDKLCLSLGSGLET----S---
Q8QVG3_ADEBA/907-1142              -----LALALGSGLAVN---------S-NN
//
        """
    )
    alignments, deletion_matrix = extract_alignments(input)
    expected_deletion_matrix = [[0] * 21 for _ in range(4)]
    expected_deletion_matrix[2][17] = 5
    expected_deletion_matrix[2][19] = 1
    expected_deletion_matrix[3][19] = 1
    assert alignments == [
        "FDNTAIAINAGKGLEFDTNTS",
        "----QLRLNIGQGLRYNP-TS",
        "------TKEDKLCLSLG----",
        "-----LALALGSGLAVN--NN",
    ]
    assert deletion_matrix == expected_deletion_matrix

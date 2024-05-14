from nanofold.data_processing.hhr_parser import parse_hhr


def test_parse_hhr(data_dir):
    with open(data_dir / "1vh2_A.hhr") as f:
        result = parse_hhr(f.readlines())
    assert len(result) == 37
    for r in result.values():
        assert r["query_start_pos"] < r["query_end_pos"]
        assert r["target_start_pos"] < r["target_end_pos"]

import pytest
from nanofold.preprocess import mmcif_parser
from nanofold.preprocess.mmcif_processor import list_available_mmcif


@pytest.fixture
def test_file(request, data_dir):
    identifiers = list_available_mmcif(data_dir)
    matched = [i for i in identifiers if request.param in i]
    assert len(matched) == 1
    return matched[0]


@pytest.mark.parametrize(
    "test_file, num_chains, num_residues, sequence",
    [
        (
            "1A00",
            4,
            141,
            "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
        ),
        (
            "115L",
            1,
            162,
            "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKVELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYK",
        ),
        (
            "1ENX",
            2,
            190,
            "XTIQPGTGYNNGYFYSYWNDGHGGVTYTNGPGGQFSVNWSNSGNFVGGKGWQPGTKNKVINFSGSYNPNGNSYLSVYGWSRNPLIEYYIVENFGTYNPSTGATKLGEVTSDGSVYDIYRTQRVNQPSIIGTATFYQYWSVRRNHRSSGSVNTANHFNAWAQQGLTLGTMDYQIVAVEGYFSSGSASITVS",
        ),
        (
            "1RNL",
            1,
            200,
            "EPATILLIDDHPMLRTGVKQLISMAPDITVVGEASNGEQGIELAESLDPDLILLDLNMPGMNGLETLDKLREKSLSGRIVVFSVSNHEEDVVTALKRGADGYLLKDMEPEDLLKALHQAAAGEMVLSEALTPVLAASLQLTPRERDILKLIAQGLPNKMIARRLDITESTVKVHVKHMLKKMKLKSRVEAAVWVHQERIF",
        ),
    ],
    indirect=["test_file"],
)
def test_parse_mmcif_file(test_file, num_chains, num_residues, sequence):
    chains = mmcif_parser.parse_mmcif_file(test_file, capture_errors=False)
    assert len(chains) == num_chains
    if len(chains) > 0:
        assert len(chains[0]["sequence"]) == num_residues
        assert len(chains[0]["translations"]) == num_residues
        assert len(chains[0]["positions"]) == num_residues
        assert chains[0]["sequence"] == sequence


@pytest.mark.parametrize(
    "test_file",
    ["1YUJ"],
    indirect=["test_file"],
)
def test_parse_mmcif_file_with_error(test_file):
    with pytest.raises(Exception):
        mmcif_parser.parse_mmcif_file(test_file, capture_errors=False)

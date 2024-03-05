import pytest
import torch
from nanofold import mmcif


@pytest.mark.parametrize(
    "model, valid_chains, num_residues, sequence",
    [
        (
            "1A00",
            2,
            141,
            "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
        ),
        ("1YUJ", 1, 54, "PKAKRAKHPPGTEKPRSRSQSEQPATCPICYAVIRQSRNLRRHLELRHFAKPGV"),
        (
            "115L",
            1,
            162,
            "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKVELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYK",
        ),
        (
            "1ENX",
            1,
            189,
            "TIQPGTGYNNGYFYSYWNDGHGGVTYTNGPGGQFSVNWSNSGNFVGGKGWQPGTKNKVINFSGSYNPNGNSYLSVYGWSRNPLIEYYIVENFGTYNPSTGATKLGEVTSDGSVYDIYRTQRVNQPSIIGTATFYQYWSVRRNHRSSGSVNTANHFNAWAQQGLTLGTMDYQIVAVEGYFSSGSASITVS",
        ),
        (
            "1GSG",
            1,
            432,
            "TNFIRQIIDEDLASGKHTTVHTRFPPEPNGYLHIGHAKSICLNFGIAQDYKGQCNLRFDDTNPVKEDIEYVESIKNDVEWLGFHWSGNVRYSSDYFDQLHAYAIELINKGLAYVDELTPEQIREYRGTLTQPGKNSPYRDRSVEENLALFEKMRAGGFEEGKACLRAKIDMASPFIVMRDPVLYRIKFAEHHQTGNKWCIYPMYDFTHCISDALEGITHSLCTLEFQDNRRLYDWVLDNITIPVHPRQYEFSRLNLEYTVMSKRKLNLLVTDKHVEGWDDPRMPTISGLRRRGYTAASIREFCKRIGVTKQDNTIEMASLESCIREDLNENAPRAMAVIDPVKLVIENYQGEGEMVTMPNHPNKPEMGSRQVPFSGEIWIDRADFREEANKQYKRLVLGKEVRLRNAYVIKAERVEKDAEGNITTIFCTYDA",
        ),
        (
            "1RNL",
            1,
            138,
            "EPATILLIDDHPMLRTGVKQLISMAPDITVVGEASNGEQGIELAESLDPDLILLDLNMPGMNGLETLDKLREKSLSGRIVVFSVSNHEEDVVTALKRGADGYLLKDMEPEDLLKALHQAAAGEMVLSEALTPVLAASL",
        ),
    ],
    indirect=["model"],
)
def test_parse_chains(model, valid_chains, num_residues, sequence):
    chains = mmcif.parse_chains(model)
    assert len(chains) == valid_chains
    assert len(chains[0]) == num_residues
    assert chains[0].sequence == sequence


@pytest.mark.parametrize(
    "model, num_chains, num_residues",
    [("1A00", 4, 141), ("1YUJ", 3, 0)],
    indirect=["model"],
)
def test_get_residue(model, num_chains, num_residues):
    chains = list(model.get_chains())
    assert len(chains) == num_chains
    residues = list(mmcif.get_residues(chains[0]))
    for r in residues:
        result = r["rotation"] @ r["rotation"].transpose(-2, -1)
        assert torch.allclose(result, torch.eye(3), atol=1e-5)
    assert len(residues) == num_residues

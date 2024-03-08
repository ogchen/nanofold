import torch
from nanofold.frame import Frame
from nanofold.model.structure import StructureModuleLayer


def test_structure_module_layer():
    single_embedding_size = 4
    len_seq = 7
    pair_embedding_size = 5
    model = StructureModuleLayer(
        single_embedding_size=single_embedding_size,
        pair_embedding_size=pair_embedding_size,
        ipa_embedding_size=6,
        ipa_num_query_points=3,
        ipa_num_value_points=3,
        ipa_num_heads=2,
        dropout=0.1,
    )
    single = torch.rand(len_seq, single_embedding_size)
    pair = torch.rand(len_seq, len_seq, pair_embedding_size)
    frames = Frame(
        rotations=torch.stack([torch.eye(3)] * len_seq), translations=torch.zeros(len_seq, 3)
    )

    s, f = model(single, pair, frames)
    assert s.shape == single.shape
    assert f.rotations.shape == frames.rotations.shape
    assert f.translations.shape == frames.translations.shape
    assert torch.allclose(f.rotations @ f.rotations.transpose(-2, -1), torch.eye(3), atol=1e-5)

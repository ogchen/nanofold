import torch
from nanofold.training.model.backbone_update import BackboneUpdate


def test_backbone_update():
    len_seq = 10
    single_embedding_size = 5
    model = BackboneUpdate(single_embedding_size)
    single = torch.rand(len_seq, single_embedding_size)
    frames = model(single)
    assert frames.rotations.shape == (len_seq, 3, 3)
    assert frames.translations.shape == (len_seq, 3)
    assert torch.allclose(
        frames.rotations @ frames.rotations.transpose(-2, -1), torch.eye(3), atol=1e-5
    )
    assert torch.allclose(frames.translations, model.linear(single)[..., 3:])


def test_backbone_update_batched():
    len_seq = 10
    single_embedding_size = 5
    model = BackboneUpdate(single_embedding_size)
    single = torch.rand(len_seq, single_embedding_size)
    frames = model(single)
    batched = model(torch.stack([single, single]))
    assert torch.allclose(frames.rotations, batched[0].rotations, atol=1e-3)
    assert torch.allclose(frames.translations, batched[0].translations, atol=1e-3)
    assert torch.allclose(frames.rotations, batched[1].rotations, atol=1e-3)
    assert torch.allclose(frames.translations, batched[1].translations, atol=1e-3)

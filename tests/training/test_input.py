from nanofold.training.model import input
import torch


def test_encode_one_hot():
    seq = "ADHIAA"
    one_hot = input.encode_one_hot(seq)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert torch.equal(one_hot, expected)


def test_input_embedder():
    embedding_size = 5
    position_bins = 3
    embedder = input.InputEmbedding(embedding_size, position_bins)

    seq = "ADHIAAAA"
    target_feat = input.encode_one_hot(seq)
    residue_index = torch.arange(len(seq))

    x = embedder(target_feat, residue_index)
    assert x.shape == (len(seq), len(seq), embedding_size)


def test_input_embedder_batched():
    embedder = input.InputEmbedding(5, 3)
    seq = "ADHIAAAA"
    target_feat = input.encode_one_hot(seq)
    residue_index = torch.arange(len(seq))
    x = embedder(target_feat, residue_index)
    batched = embedder(
        torch.stack([target_feat, target_feat]), torch.stack([residue_index, residue_index])
    )
    assert torch.allclose(x, batched[0], atol=1e-3)
    assert torch.allclose(x, batched[1], atol=1e-3)

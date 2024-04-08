import torch

from nanofold.training.chain_dataset import encode_msa
from nanofold.training.chain_dataset import encode_one_hot
from nanofold.training.chain_dataset import preprocess_msa
from nanofold.training.model import input


def test_input_embedder():
    pair_embedding_size = 5
    msa_embedding_size = 10
    position_bins = 3
    embedder = input.InputEmbedding(pair_embedding_size, msa_embedding_size, position_bins)

    num_msa = 4
    seq = "ADHIAAAA"
    target_feat = encode_one_hot(seq)
    msa = {"alignments": [seq], "deletion_matrix": [torch.zeros(len(seq))]}
    alignments_one_hot, deletion_feat = encode_msa(preprocess_msa(msa, num_msa))
    msa_feat = torch.cat((alignments_one_hot, deletion_feat), dim=-1)
    residue_index = torch.arange(len(seq))

    msa_rep, pair_rep = embedder(target_feat, residue_index, msa_feat)
    assert pair_rep.shape == (len(seq), len(seq), pair_embedding_size)
    assert msa_rep.shape == (num_msa, len(seq), msa_embedding_size)


def test_input_embedder_batched():
    embedder = input.InputEmbedding(5, 10, 3)

    seq = "ADHIAAAA"
    target_feat = encode_one_hot(seq)
    msa = {"alignments": [seq], "deletion_matrix": [torch.zeros(len(seq))]}
    alignments_one_hot, deletion_feat = encode_msa(preprocess_msa(msa, 4))
    msa_feat = torch.cat((alignments_one_hot, deletion_feat), dim=-1)
    residue_index = torch.arange(len(seq))

    msa_rep, pair_rep = embedder(target_feat, residue_index, msa_feat)
    batched_msa, batched_pair = embedder(
        torch.stack([target_feat, target_feat]),
        torch.stack([residue_index, residue_index]),
        torch.stack([msa_feat, msa_feat]),
    )
    assert torch.allclose(pair_rep, batched_pair[0], atol=1e-3)
    assert torch.allclose(pair_rep, batched_pair[1], atol=1e-3)
    assert torch.allclose(msa_rep, batched_msa[0], atol=1e-3)
    assert torch.allclose(msa_rep, batched_msa[1], atol=1e-3)

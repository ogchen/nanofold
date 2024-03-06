import torch
from nanofold.invariant_point_attention import InvariantPointAttention
from nanofold.frame import Frame


@torch.no_grad
def test_invariant_point_attention():
    len_seq = 10
    pair_embedding_size = 7
    single_embedding_size = 3
    embedding_size = 5
    query_points = 4
    value_points = 8
    num_heads = 2
    model = InvariantPointAttention(
        pair_embedding_size=pair_embedding_size,
        single_embedding_size=single_embedding_size,
        embedding_size=embedding_size,
        num_query_points=query_points,
        num_value_points=value_points,
        num_heads=num_heads,
    )
    single_representation = torch.stack(
        [torch.arange(single_embedding_size).float()] * len_seq
    )
    pair_representation = torch.zeros(len_seq, len_seq, pair_embedding_size)
    frames = Frame(
        rotations=torch.stack([torch.eye(3)] * len_seq),
        translations=torch.stack([torch.zeros(3)] * len_seq),
    )

    result = model(single_representation, pair_representation, frames)
    assert result.shape == (len_seq, single_embedding_size)

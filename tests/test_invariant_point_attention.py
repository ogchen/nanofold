import math
import torch
from nanofold.invariant_point_attention import InvariantPointAttention
from nanofold.frame import Frame


class TestInvariantPointAttention:
    def setup_method(self):
        self.len_seq = 10
        self.pair_embedding_size = 7
        self.single_embedding_size = 3
        self.embedding_size = 5
        self.query_points = 4
        self.value_points = 8
        self.num_heads = 2
        self.model = InvariantPointAttention(
            pair_embedding_size=self.pair_embedding_size,
            single_embedding_size=self.single_embedding_size,
            embedding_size=self.embedding_size,
            num_query_points=self.query_points,
            num_value_points=self.value_points,
            num_heads=self.num_heads,
        )
        self.single_representation = torch.stack(
            [torch.arange(self.single_embedding_size).float()] * self.len_seq
        )
        self.pair_representation = torch.zeros(
            self.len_seq, self.len_seq, self.pair_embedding_size
        )
        self.frames = Frame(
            rotations=torch.stack([torch.eye(3)] * self.len_seq),
            translations=torch.stack([torch.zeros(3)] * self.len_seq),
        )

    @torch.no_grad
    def test_shape(self):
        result = self.model(
            self.single_representation, self.pair_representation, self.frames
        )
        assert result.shape == (self.len_seq, self.single_embedding_size)

    @torch.no_grad
    def test_invariant_to_transformations(self):
        rotation = torch.tensor([[math.cos(1), -math.sin(1), 0], [math.sin(1), math.cos(1), 0], [0, 0, 1]])
        transform = Frame(rotations=rotation, translations=torch.ones(3))
        attention = self.model(
            self.single_representation, self.pair_representation, self.frames
        )
        transformed_attention = self.model(
            self.single_representation, self.pair_representation, Frame.compose(transform, self.frames)
        )

        assert torch.allclose(attention, transformed_attention)

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
        self.num_query_points = 4
        self.num_value_points = 8
        self.num_heads = 2
        self.model = InvariantPointAttention(
            pair_embedding_size=self.pair_embedding_size,
            single_embedding_size=self.single_embedding_size,
            embedding_size=self.embedding_size,
            num_query_points=self.num_query_points,
            num_value_points=self.num_value_points,
            num_heads=self.num_heads,
        )
        self.single_representation = (
            torch.arange(self.single_embedding_size * self.len_seq)
            .float()
            .reshape(self.len_seq, self.single_embedding_size)
        )
        self.pair_representation = (
            torch.arange(self.len_seq**2 * self.pair_embedding_size)
            .float()
            .reshape(self.len_seq, self.len_seq, self.pair_embedding_size)
        )
        rotations = torch.tensor(
            [
                [
                    [math.cos(i), -math.sin(i), 0],
                    [math.sin(i), math.cos(i), 0],
                    [0, 0, 1],
                ]
                for i in range(self.len_seq)
            ]
        )
        translations = torch.arange(self.len_seq * 3).float().reshape(self.len_seq, 3)
        self.frames = Frame(
            rotations=rotations,
            translations=translations,
        )

    @torch.no_grad
    def test_shape(self):
        result = self.model(
            self.single_representation, self.pair_representation, self.frames
        )
        assert result.shape == (self.len_seq, self.single_embedding_size)

    @torch.no_grad
    def test_invariant_to_transformations(self):
        rotation = torch.tensor(
            [[math.cos(1), -math.sin(1), 0], [math.sin(1), math.cos(1), 0], [0, 0, 1]]
        )
        transform = Frame(rotations=rotation, translations=torch.ones(3))
        attention = self.model(
            self.single_representation, self.pair_representation, self.frames
        )
        transformed_attention = self.model(
            self.single_representation,
            self.pair_representation,
            Frame.compose(transform, self.frames),
        )
        assert torch.allclose(attention, transformed_attention, atol=1e-5)

    @torch.no_grad
    def test_single_rep_weight(self):
        weight = self.model.single_rep_weight(self.single_representation)
        assert weight.shape == (self.num_heads, self.len_seq, self.len_seq)

        q = torch.stack([f(self.single_representation) for f in self.model.query])
        k = torch.stack([f(self.single_representation) for f in self.model.key])
        for h in range(weight.shape[0]):
            for i in range(weight.shape[1]):
                for j in range(weight.shape[2]):
                    assert torch.allclose(
                        weight[h, i, j],
                        math.sqrt(1 / self.embedding_size)
                        * torch.dot(q[h, i], k[h, j]),
                        atol=1e-5,
                    )

    @torch.no_grad
    def test_frame_weight(self):
        for i in range(self.num_heads):
            self.model.scale_head[i] = i + 1
        weight = self.model.frame_weight(self.frames, self.single_representation)
        assert weight.shape == (self.num_heads, self.len_seq, self.len_seq)

        qp = torch.stack(
            [f(self.single_representation) for f in self.model.query_points]
        ).reshape(self.num_heads, self.num_query_points, -1, 3)
        kp = torch.stack(
            [f(self.single_representation) for f in self.model.key_points]
        ).reshape(self.num_heads, self.num_query_points, -1, 3)

        for h in range(weight.shape[0]):
            for i in range(weight.shape[1]):
                for j in range(weight.shape[2]):
                    sum_distance = 0
                    for p in range(self.num_query_points):
                        diff = Frame.apply(
                            self.frames[i], qp[h][p][i].unsqueeze(0)
                        ) - Frame.apply(self.frames[j], kp[h][p][j].unsqueeze(0))
                        sum_distance += torch.linalg.vector_norm(diff) ** 2
                    scale_factor = -self.model.softplus(self.model.scale_head[h]) * math.sqrt(2/(9 * self.num_query_points)) * 1/2
                    assert torch.allclose(scale_factor * sum_distance, weight[h, i, j], atol=1e-5)

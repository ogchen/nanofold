import math
import torch
from torch import nn
from nanofold.frame import Frame


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        pair_embedding_size,
        single_embedding_size,
        embedding_size,
        num_query_points,
        num_value_points,
        num_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.embedding_size = embedding_size
        self.query = nn.ModuleList(
            [
                nn.Linear(single_embedding_size, embedding_size, bias=False)
                for _ in range(self.num_heads)
            ]
        )
        self.key = nn.ModuleList(
            [
                nn.Linear(single_embedding_size, embedding_size, bias=False)
                for _ in range(self.num_heads)
            ]
        )
        self.value = nn.ModuleList(
            [
                nn.Linear(single_embedding_size, embedding_size, bias=False)
                for _ in range(self.num_heads)
            ]
        )
        self.query_points = nn.ModuleList(
            [
                nn.Linear(single_embedding_size, 3, bias=False)
                for _ in range(self.num_heads * self.num_query_points)
            ]
        )
        self.key_points = nn.ModuleList(
            [
                nn.Linear(single_embedding_size, 3, bias=False)
                for _ in range(self.num_heads * self.num_query_points)
            ]
        )
        self.value_points = nn.ModuleList(
            [
                nn.Linear(single_embedding_size, 3, bias=False)
                for _ in range(self.num_heads * self.num_value_points)
            ]
        )
        self.bias = nn.ModuleList(
            [
                nn.Linear(pair_embedding_size, 1, bias=False)
                for _ in range(self.num_heads)
            ]
        )
        self.out = nn.Linear(
            self.num_heads
            * (pair_embedding_size + embedding_size + self.num_value_points * (3 + 1)),
            single_embedding_size,
        )
        self.softplus = nn.Softplus()
        self.scale_head = nn.Parameter(torch.ones(self.num_heads))

    def single_rep_weight(self, single_representation):
        q = torch.stack([f(single_representation) for f in self.query])
        k = torch.stack([f(single_representation) for f in self.key])
        weight = math.sqrt(1 / self.embedding_size) * q @ k.transpose(-2, -1)
        return weight

    def pair_rep_weight(self, pair_representation):
        weight = torch.stack([f(pair_representation) for f in self.bias]).squeeze(-1)
        return weight

    def frame_weight(self, frames, single_representation):
        qp = torch.stack([f(single_representation) for f in self.query_points]).reshape(
            self.num_heads, self.num_query_points, -1, 3
        )
        kp = torch.stack([f(single_representation) for f in self.key_points]).reshape(
            self.num_heads, self.num_query_points, -1, 3
        )

        local_qp = Frame.apply(frames, qp.transpose(-2, -3)).transpose(-2, -3)
        local_kp = Frame.apply(frames, kp.transpose(-2, -3)).transpose(-2, -3)
        difference = local_qp.unsqueeze(-2) - local_kp.unsqueeze(-3)
        squared_distance = difference.unsqueeze(-2) @ difference.unsqueeze(-1)
        squared_distance = squared_distance.squeeze(-1, -2)
        weight = torch.sum(squared_distance, dim=-3)
        weight = self.softplus(self.scale_head) * weight.transpose(0, -1)
        weight = (
            -1 * math.sqrt(1 / (18 * self.num_query_points)) * weight.transpose(0, -1)
        )
        return weight

    def single_rep_attention(self, weight, single_representation):
        len_seq, _ = single_representation.shape
        v = torch.stack([f(single_representation) for f in self.value])
        attention = weight.unsqueeze(-3) @ v.unsqueeze(-3)
        attention = attention.squeeze(-3)
        attention = attention.transpose(0, 1).reshape(len_seq, -1)
        return attention

    def pair_rep_attention(self, weight, pair_representation):
        len_seq, _, _ = pair_representation.shape
        attention = weight.unsqueeze(-3) @ pair_representation
        attention = torch.sum(attention, dim=-2)
        attention = attention.transpose(0, 1).reshape(len_seq, -1)
        return attention

    def frame_attention(self, weight, frames, single_representation):
        len_seq, _ = single_representation.shape
        vp = torch.stack([f(single_representation) for f in self.value_points]).reshape(
            self.num_heads, self.num_value_points, -1, 3
        )
        local_vp = Frame.apply(frames, vp.transpose(-2, -3)).transpose(-2, -3)
        local_out_points = weight.unsqueeze(-3) @ local_vp
        inverse_frames = Frame.inverse(frames)
        global_out_points = Frame.apply(
            inverse_frames, local_out_points.transpose(-2, -3)
        ).transpose(-2, -3)

        out_points = global_out_points.transpose(0, -2).reshape(len_seq, -1)
        out_points_norm = (
            torch.linalg.vector_norm(global_out_points, dim=-1)
            .transpose(0, 2)
            .reshape(len_seq, -1)
        )
        return torch.cat([out_points, out_points_norm], dim=-1)

    def forward(self, single_representation, pair_representation, frames):
        weight = (
            self.single_rep_weight(single_representation)
            + self.pair_rep_weight(pair_representation)
            + self.frame_weight(frames, single_representation)
        )
        weight = nn.functional.softmax(weight, dim=-1)

        attention = torch.cat(
            [
                self.single_rep_attention(weight, single_representation),
                self.pair_rep_attention(weight, pair_representation),
                self.frame_attention(weight, frames, single_representation),
            ],
            dim=-1,
        )
        attention = self.out(attention)
        return attention

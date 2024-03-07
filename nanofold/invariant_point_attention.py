import math
import torch
from torch import nn
from nanofold.frame import Frame


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        single_embedding_size,
        pair_embedding_size,
        embedding_size,
        num_query_points,
        num_value_points,
        num_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.query = nn.Linear(
            single_embedding_size, embedding_size * num_heads, bias=False
        )
        self.key = nn.Linear(
            single_embedding_size, embedding_size * num_heads, bias=False
        )
        self.value = nn.Linear(
            single_embedding_size, embedding_size * num_heads, bias=False
        )
        self.query_points = nn.Linear(
            single_embedding_size, 3 * num_query_points * num_heads, bias=False
        )
        self.key_points = nn.Linear(
            single_embedding_size, 3 * num_query_points * num_heads, bias=False
        )
        self.value_points = nn.Linear(
            single_embedding_size, 3 * num_value_points * num_heads, bias=False
        )
        self.bias = nn.Linear(pair_embedding_size, num_heads, bias=False)
        self.out = nn.Linear(
            self.num_heads
            * (pair_embedding_size + embedding_size + self.num_value_points * (3 + 1)),
            single_embedding_size,
        )
        self.softplus = nn.Softplus()
        self.scale_head = nn.Parameter(torch.ones(self.num_heads))
        self.scale_single_rep = 1 / math.sqrt(embedding_size)
        self.scale_frame = -1 / math.sqrt(18 * self.num_query_points)

    def single_rep_weight(self, single_representation):
        len_seq = single_representation.shape[0]
        q = (
            self.query(single_representation)
            .view(len_seq, self.num_heads, -1)
            .transpose(0, 1)
        )
        k = (
            self.key(single_representation)
            .view(len_seq, self.num_heads, -1)
            .transpose(0, 1)
        )
        weight = self.scale_single_rep * q @ k.transpose(-2, -1)
        return weight

    def pair_rep_weight(self, pair_representation):
        weight = self.bias(pair_representation).permute(2, 0, 1)
        return weight

    def frame_weight(self, frames, single_representation):
        len_seq = single_representation.shape[0]
        qp = (
            self.query_points(single_representation)
            .view(len_seq, self.num_heads, -1, 3)
            .transpose(0, 1)
        )
        kp = (
            self.key_points(single_representation)
            .view(len_seq, self.num_heads, -1, 3)
            .transpose(0, 1)
        )
        local_qp = Frame.apply(frames, qp)
        local_kp = Frame.apply(frames, kp)
        difference = local_qp.unsqueeze(-3) - local_kp.unsqueeze(-4)
        squared_distance = difference.unsqueeze(-2) @ difference.unsqueeze(-1)
        squared_distance = squared_distance.squeeze(-1, -2)
        weight = torch.sum(squared_distance, dim=-1)
        weight = self.softplus(self.scale_head) * weight.transpose(0, -1)
        weight = self.scale_frame * weight.transpose(0, -1)
        return weight

    def single_rep_attention(self, weight, single_representation):
        len_seq = single_representation.shape[0]
        v = (
            self.value(single_representation)
            .view(len_seq, self.num_heads, -1)
            .transpose(0, 1)
        )
        attention = weight.unsqueeze(-3) @ v.unsqueeze(-3)
        attention = attention.squeeze(-3)
        attention = attention.transpose(0, 1).reshape(len_seq, -1)
        return attention

    def pair_rep_attention(self, weight, pair_representation):
        len_seq = pair_representation.shape[0]
        attention = weight.unsqueeze(-3) @ pair_representation
        attention = torch.sum(attention, dim=-2)
        attention = attention.transpose(0, 1).reshape(len_seq, -1)
        return attention

    def frame_attention(self, weight, frames, single_representation):
        len_seq = single_representation.shape[0]
        vp = (
            self.value_points(single_representation)
            .view(len_seq, self.num_heads, -1, 3)
            .transpose(0, 1)
        )
        local_vp = Frame.apply(frames, vp)
        local_out_points = weight.unsqueeze(-3) @ local_vp.transpose(-2, -3)
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

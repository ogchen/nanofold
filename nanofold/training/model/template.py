import torch.nn as nn

from nanofold.training.model.pair_transition import PairTransition
from nanofold.training.model.triangular_attention import TriangleAttentionStartingNode
from nanofold.training.model.triangular_attention import TriangleAttentionEndingNode
from nanofold.training.model.triangular_update import TriangleMultiplicationOutgoing
from nanofold.training.model.triangular_update import TriangleMultiplicationIncoming
from nanofold.training.model.util import DropoutByDimension
from nanofold.training.model.util import LinearWithView


class TemplatePairStackBlock(nn.Module):
    def __init__(
        self,
        template_embedding_size,
        channels,
        heads,
        transition_multiplier,
        device,
        p_dropout=0.25,
    ):
        super().__init__()
        self.dropout = DropoutByDimension(p_dropout, device)
        self.triangle_update_outgoing = TriangleMultiplicationOutgoing(
            template_embedding_size, channels
        )
        self.triangle_update_incoming = TriangleMultiplicationIncoming(
            template_embedding_size, channels
        )
        self.triangle_attention_starting = TriangleAttentionStartingNode(
            template_embedding_size, heads, channels
        )
        self.triangle_attention_ending = TriangleAttentionEndingNode(
            template_embedding_size, heads, channels
        )
        self.pair_transition = PairTransition(template_embedding_size, transition_multiplier)
        self.layer_norm = nn.LayerNorm(template_embedding_size)

    def forward(self, template_rep):
        template_rep = template_rep + self.dropout(
            self.triangle_attention_starting(template_rep), dim=-3
        )
        template_rep = template_rep + self.dropout(
            self.triangle_attention_ending(template_rep), dim=-2
        )
        template_rep = template_rep + self.dropout(
            self.triangle_update_outgoing(template_rep), dim=-3
        )
        template_rep = template_rep + self.dropout(
            self.triangle_update_incoming(template_rep), dim=-3
        )
        template_rep = template_rep + self.pair_transition(template_rep)
        return self.layer_norm(template_rep)


class TemplatePairStack(nn.Module):
    def __init__(
        self,
        template_embedding_size,
        channels,
        heads,
        num_blocks,
        transition_multiplier,
        device,
        template_input_size=84,
    ):
        super().__init__()
        self.linear = nn.Linear(template_input_size, channels)
        self.blocks = nn.Sequential(
            *[
                TemplatePairStackBlock(
                    template_embedding_size,
                    channels,
                    heads,
                    transition_multiplier,
                    device,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, template_pair_feat):
        template_rep = self.linear(template_pair_feat)
        template_rep = self.blocks(template_rep)
        return template_rep


class TemplatePointwiseAttention(nn.Module):
    def __init__(self, pair_embedding_size, template_embedding_size, num_heads, num_channels):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.query = LinearWithView(pair_embedding_size, (num_heads, num_channels), bias=False)
        self.key = LinearWithView(template_embedding_size, (num_heads, num_channels), bias=False)
        self.value = LinearWithView(template_embedding_size, (num_heads, num_channels), bias=False)
        self.projection = nn.Linear(num_heads * num_channels, pair_embedding_size)

    def forward(self, template_rep, pair_rep):
        q = self.query(pair_rep)
        k = self.key(template_rep)
        v = self.value(template_rep)

        weights = (q.unsqueeze(-2) @ k.movedim(-5, -1)) / (self.num_channels**0.5)
        weights = nn.functional.softmax(weights, dim=-1)
        attention = weights @ v.movedim(-5, -2)
        return self.projection(attention.flatten(start_dim=-3))

from torch import nn
from nanofold.model.backbone_update import BackboneUpdate
from nanofold.frame import Frame
from nanofold.model.invariant_point_attention import InvariantPointAttention


class StructureModuleLayer(nn.Module):
    def __init__(
        self,
        single_embedding_size,
        pair_embedding_size,
        ipa_embedding_size,
        ipa_num_query_points,
        ipa_num_value_points,
        ipa_num_heads,
        dropout,
    ):
        super().__init__()
        self.invariant_point_attention = InvariantPointAttention(
            single_embedding_size=single_embedding_size,
            pair_embedding_size=pair_embedding_size,
            ipa_embedding_size=ipa_embedding_size,
            num_query_points=ipa_num_query_points,
            num_value_points=ipa_num_value_points,
            num_heads=ipa_num_heads,
        )
        self.backbone_update = BackboneUpdate(single_embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(single_embedding_size)
        self.layer_norm2 = nn.LayerNorm(single_embedding_size)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(single_embedding_size, single_embedding_size)
        self.linear2 = nn.Linear(single_embedding_size, single_embedding_size)
        self.linear3 = nn.Linear(single_embedding_size, single_embedding_size)

    @classmethod
    def from_config(cls, config):
        return cls(
            single_embedding_size=config.getint("Other", "single_embedding_size"),
            pair_embedding_size=config.getint("InputEmbedding", "pair_embedding_size"),
            ipa_embedding_size=config.getint("InvariantPointAttention", "embedding_size"),
            ipa_num_query_points=config.getint("InvariantPointAttention", "num_query_points"),
            ipa_num_value_points=config.getint("InvariantPointAttention", "num_value_points"),
            ipa_num_heads=config.getint("InvariantPointAttention", "num_heads"),
            dropout=config.getfloat("StructureModule", "dropout"),
        )

    def forward(self, single, pair, frames):
        single += self.invariant_point_attention(single, pair, frames)
        single = self.layer_norm1(self.dropout(single))
        single += self.linear3(self.relu(self.linear2(self.relu(self.linear1(single)))))
        single = self.layer_norm2(self.dropout(single))
        frames = Frame.compose(frames, self.backbone_update(single))
        return single, frames

import torch.nn as nn


class PairTransition(nn.Sequential):
    def __init__(self, pair_embedding_size, transition_multiplier):
        super().__init__(
            nn.LayerNorm(pair_embedding_size),
            nn.Linear(pair_embedding_size, pair_embedding_size * transition_multiplier),
            nn.ReLU(),
            nn.Linear(pair_embedding_size * transition_multiplier, pair_embedding_size),
        )

import math
import torch.nn as nn
import torch

from nanofold.training.model.relative_position_encoding import RelativePositionEncoding
from nanofold.training.model.transition import Transition


def fourier_embedding(t, c):
    normal = torch.distributions.MultivariateNormal(torch.zeros(c), torch.eye(c))
    w, b = normal.sample([2]).to(t.device)
    return torch.cos(2 * math.pi * (t * w + b))


class DiffusionConditioning(nn.Module):
    def __init__(
        self,
        position_bins,
        stacked_single_embedding_size,
        pair_embedding_size,
        stacked_pair_embedding_size,
        fourier_embedding_size,
        data_std_dev,
        transition_multiplier=2,
    ):
        super().__init__()
        self.data_std_dev = data_std_dev
        self.fourier_embedding_size = fourier_embedding_size
        self.relative_position_encoding = RelativePositionEncoding(
            position_bins, pair_embedding_size
        )
        self.pair = nn.Sequential(
            nn.LayerNorm(stacked_pair_embedding_size),
            nn.Linear(stacked_pair_embedding_size, stacked_pair_embedding_size, bias=False),
        )
        self.pair_transition = nn.ModuleList(
            [
                Transition(stacked_pair_embedding_size, transition_multiplier),
                Transition(stacked_pair_embedding_size, transition_multiplier),
            ]
        )
        self.single = nn.Sequential(
            nn.LayerNorm(stacked_single_embedding_size),
            nn.Linear(stacked_single_embedding_size, stacked_single_embedding_size, bias=False),
        )
        self.single_transition = nn.ModuleList(
            [
                Transition(stacked_single_embedding_size, transition_multiplier),
                Transition(stacked_single_embedding_size, transition_multiplier),
            ]
        )

        self.n_embedder = nn.Sequential(
            nn.LayerNorm(fourier_embedding_size),
            nn.Linear(fourier_embedding_size, stacked_single_embedding_size, bias=False),
        )

    def forward(self, t, features, input, trunk, pair_rep):
        pair_rep = torch.concat(
            [pair_rep, self.relative_position_encoding(features["residue_index"])], dim=-1
        )
        pair_rep = self.pair(pair_rep)
        for transition in self.pair_transition:
            pair_rep = pair_rep + transition(pair_rep)
        single = torch.concat([input, trunk], dim=-1)
        single = self.single(single)
        n = fourier_embedding(0.25 * torch.log(t / self.data_std_dev), self.fourier_embedding_size)
        single = single + self.n_embedder(n)
        for transition in self.single_transition:
            single = single + transition(single)
        return single, pair_rep

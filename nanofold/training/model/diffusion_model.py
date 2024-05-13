import torch
import torch.nn as nn
import torch.nn.functional as F

from nanofold.training.loss import compute_smooth_lddt_loss
from nanofold.training.model.atom_attention_decoder import AtomAttentionDecoder
from nanofold.training.model.atom_attention_encoder import AtomAttentionEncoder
from nanofold.training.model.diffusion_conditioning import DiffusionConditioning
from nanofold.training.model.diffusion_transformer import DiffusionTransformer
from nanofold.training.util import uniform_random_rotation
from nanofold.training.util import rigid_align


class DiffusionModel(nn.Module):
    def __init__(
        self,
        batch_size,
        steps,
        atom_embedding_size,
        atom_pair_embedding_size,
        token_embedding_size,
        input_embedding_size,
        single_embedding_size,
        pair_embedding_size,
        fourier_embedding_size,
        num_atom_transformer_blocks,
        num_atom_transformer_heads,
        num_atom_transformer_queries,
        num_atom_transformer_keys,
        num_diffusion_transformer_blocks,
        num_diffusion_transformer_heads,
        position_bins,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        data_std_dev=16,
        s_max=160,
        s_min=0.0004,
        p=7,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.normal = torch.distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.data_std_dev = data_std_dev
        stacked_single_embedding_size = input_embedding_size + single_embedding_size
        stacked_pair_embedding_size = 2 * pair_embedding_size
        self.diffusion_conditioning = DiffusionConditioning(
            position_bins,
            stacked_single_embedding_size,
            pair_embedding_size,
            stacked_pair_embedding_size,
            fourier_embedding_size,
            data_std_dev,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_embedding_size,
            atom_pair_embedding_size,
            token_embedding_size,
            single_embedding_size,
            stacked_pair_embedding_size,
            num_atom_transformer_blocks,
            num_atom_transformer_heads,
            num_atom_transformer_queries,
            num_atom_transformer_keys,
        )
        self.atom_attention_decoder = AtomAttentionDecoder(
            token_embedding_size,
            atom_embedding_size,
            atom_pair_embedding_size,
            token_embedding_size,
            num_atom_transformer_blocks,
            num_atom_transformer_heads,
            num_atom_transformer_queries,
            num_atom_transformer_keys,
        )
        self.diffusion_transformer = DiffusionTransformer(
            token_embedding_size,
            stacked_single_embedding_size,
            stacked_pair_embedding_size,
            num_diffusion_transformer_blocks,
            num_diffusion_transformer_heads,
        )
        self.single_embedder = nn.Sequential(
            nn.LayerNorm(stacked_single_embedding_size),
            nn.Linear(stacked_single_embedding_size, token_embedding_size, bias=False),
        )
        self.layer_norm = nn.LayerNorm(token_embedding_size)
        self.schedule = (
            data_std_dev
            * (
                s_max ** (1 / p)
                + torch.arange(0, 1, 1 / steps) * (s_min ** (1 / p) - s_max ** (1 / p))
            )
            ** p
        )

    def centre_random_augmentation(self, x):
        batch_dims = x.shape[:-2]
        x = x - x.mean(dim=-2, keepdim=True)
        rotation = uniform_random_rotation(*batch_dims).to(x.device)
        translation = self.normal.sample(batch_dims).to(x.device)
        x = (rotation.unsqueeze(-3) @ x.unsqueeze(-1)).squeeze(-1) + translation.unsqueeze(-2)
        return x

    def diffusion(self, x_noisy, t, features, input, trunk, pair_rep):
        stacked_single, stacked_pair = self.diffusion_conditioning(
            t, features, input, trunk, pair_rep
        )
        r = x_noisy * (t**2 + self.data_std_dev**2) ** -0.5
        a, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            features["ref_pos"], features["ref_space_uid"], r, trunk, stacked_pair
        )
        a = a + self.single_embedder(stacked_single)
        a = self.diffusion_transformer(a, stacked_single, stacked_pair, beta=None)
        a = self.layer_norm(a)
        r_update = self.atom_attention_decoder(a, q_skip, c_skip, p_skip)
        x_out = (
            x_noisy * self.data_std_dev**2 / (self.data_std_dev**2 + t**2)
            + r_update * self.data_std_dev * t / (self.data_std_dev**2 + t**2) ** 0.5
        )
        return x_out

    def sample_diffusion(self, features, input, trunk, pair_rep):
        schedule = self.schedule.to(trunk.device)
        x = schedule[0] * self.normal.sample(features["local_coords"].shape[:-1]).flatten(
            start_dim=-3, end_dim=-2
        ).to(trunk.device)

        for c_prev, c in zip(schedule[:-1], schedule[1:]):
            x = self.centre_random_augmentation(x)
            gamma = self.gamma_0 if c > self.gamma_min else 0
            t = c_prev * (gamma + 1)
            noise = (
                self.noise_scale
                * ((t**2 - c_prev**2) ** 0.5)
                * self.normal.sample(x.shape[:-1]).to(x.device)
            )
            x_noisy = x + noise
            x_denoised = self.diffusion(x_noisy, t, features, input, trunk, pair_rep)
            delta = (x - x_denoised) / t
            dt = c - t
            x = x_noisy + self.step_scale * delta * dt

        return x

    def train_diffusion(self, features, input, trunk, pair_rep):
        x_gt = torch.tile(features["coords_truth"].unsqueeze(-3), (self.batch_size, 1, 1))
        x_gt = self.centre_random_augmentation(x_gt)
        t = self.data_std_dev * torch.exp(-1.2 + 1.5 * torch.normal(0, 1, (self.batch_size, 1, 1)))
        t = t.to(x_gt.device)
        x_noisy = x_gt + (t * self.normal.sample(x_gt.shape[:-1]).to(x_gt.device))

        x = self.diffusion(x_noisy, t, features, input, trunk, pair_rep)

        with torch.no_grad():
            x_gt_aligned = rigid_align(x_gt, x).detach()
        mse_loss = (
            F.mse_loss(x, x_gt_aligned, reduction="none").mean(dim=(-2, -1), keepdim=True) / 3
        )
        smooth_lddt_loss = compute_smooth_lddt_loss(x, x_gt_aligned)
        diffusion_loss = (t**2 + self.data_std_dev**2) / (t + self.data_std_dev) ** 2 * (
            mse_loss
        ) + smooth_lddt_loss
        return diffusion_loss.mean()

    def forward(self, features, input, trunk, pair_rep):
        return self.train_diffusion(features, input, trunk, pair_rep)

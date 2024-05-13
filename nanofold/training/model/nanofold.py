import torch
from torch import nn

from nanofold.training.loss import DistogramLoss
from nanofold.training.model.diffusion_model import DiffusionModel
from nanofold.training.model.input import InputEmbedding
from nanofold.training.model.msa_module import MSAModule
from nanofold.training.model.pairformer import Pairformer
from nanofold.training.model.relative_position_encoding import RelativePositionEncoding
from nanofold.training.model.template_embedder import TemplateEmbedder


class Nanofold(nn.Module):
    def __init__(
        self,
        device,
        compile_model,
        use_grad_checkpoint,
        num_recycle,
        single_embedding_size,
        pair_embedding_size,
        input_atom_embedding_size,
        input_atom_pair_embedding_size,
        input_token_embedding_size,
        position_bins,
        num_atom_transformer_blocks,
        num_atom_transformer_heads,
        num_atom_transformer_queries,
        num_atom_transformer_keys,
        product_embedding_size,
        num_msa_samples,
        num_msa_blocks,
        msa_embedding_size,
        msa_averaging_embedding_size,
        num_msa_heads,
        msa_transition_multiplier,
        num_triangular_update_channels,
        num_triangular_attention_channels,
        num_triangular_attention_heads,
        num_template_blocks,
        template_embedding_size,
        num_pairformer_blocks,
        num_pair_heads,
        pairformer_transition_multiplier,
        diffusion_steps,
        diffusion_batch_size,
        atom_embedding_size,
        atom_pair_embedding_size,
        token_embedding_size,
        num_diffusion_transformer_blocks,
        num_diffusion_transformer_heads,
        fourier_embedding_size,
        num_distogram_bins,
    ):
        super().__init__()

        self.device = device
        self.use_grad_checkpoint = use_grad_checkpoint
        self.num_recycle = num_recycle
        self.input_embedder = InputEmbedding(
            input_atom_embedding_size,
            input_atom_pair_embedding_size,
            input_token_embedding_size,
            single_embedding_size,
            pair_embedding_size,
            num_atom_transformer_blocks,
            num_atom_transformer_heads,
            num_atom_transformer_queries,
            num_atom_transformer_keys,
        )
        input_embedding_size = self.input_embedder.embedding_size
        self.linear_input_single = nn.Linear(input_embedding_size, single_embedding_size)
        self.linear_input_pair_a = nn.Linear(input_embedding_size, pair_embedding_size)
        self.linear_input_pair_b = nn.Linear(input_embedding_size, pair_embedding_size)
        self.relative_position_encoding = RelativePositionEncoding(
            position_bins, pair_embedding_size
        )
        self.transition_pair = nn.Sequential(
            nn.LayerNorm(pair_embedding_size),
            nn.Linear(pair_embedding_size, pair_embedding_size, bias=False),
        )
        self.template_embedder = TemplateEmbedder(
            template_embedding_size,
            pair_embedding_size,
            num_triangular_update_channels,
            num_triangular_attention_channels,
            num_triangular_attention_heads,
            num_pair_heads,
            pairformer_transition_multiplier,
            num_template_blocks,
        )
        self.msa_module = MSAModule(
            num_msa_blocks,
            input_embedding_size,
            msa_embedding_size,
            num_msa_samples,
            pair_embedding_size,
            msa_averaging_embedding_size,
            product_embedding_size,
            num_triangular_update_channels,
            num_triangular_attention_channels,
            num_msa_heads,
            num_pair_heads,
            msa_transition_multiplier,
        )
        self.transition_single = nn.Sequential(
            nn.LayerNorm(single_embedding_size),
            nn.Linear(single_embedding_size, single_embedding_size, bias=False),
        )
        self.pairformer = Pairformer(
            single_embedding_size,
            pair_embedding_size,
            num_triangular_update_channels,
            num_triangular_attention_channels,
            num_triangular_attention_heads,
            num_pair_heads,
            pairformer_transition_multiplier,
            num_pairformer_blocks,
        )
        self.diffusion_model = DiffusionModel(
            diffusion_batch_size,
            diffusion_steps,
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
        )
        self.distogram_loss = DistogramLoss(pair_embedding_size, num_distogram_bins, device)
        self.compile(compile_model)

    def compile(self, compile_model):
        self.pairformer = torch.compile(
            self.pairformer,
            disable=not compile_model,
            dynamic=True,
        )

    @staticmethod
    def get_args(config):
        return {
            "device": config["device"],
            "compile_model": config["compile_model"],
            "use_grad_checkpoint": config["use_grad_checkpoint"],
            "num_recycle": config["num_recycle"],
            "single_embedding_size": config["single_embedding_size"],
            "pair_embedding_size": config["pair_embedding_size"],
            "input_atom_embedding_size": config["input_atom_embedding_size"],
            "input_atom_pair_embedding_size": config["input_atom_pair_embedding_size"],
            "input_token_embedding_size": config["input_token_embedding_size"],
            "position_bins": config["position_bins"],
            "num_atom_transformer_blocks": config["num_atom_transformer_blocks"],
            "num_atom_transformer_heads": config["num_atom_transformer_heads"],
            "num_atom_transformer_queries": config["num_atom_transformer_queries"],
            "num_atom_transformer_keys": config["num_atom_transformer_keys"],
            "product_embedding_size": config["product_embedding_size"],
            "num_msa_samples": config["num_msa_samples"],
            "num_msa_blocks": config["num_msa_blocks"],
            "msa_embedding_size": config["msa_embedding_size"],
            "msa_averaging_embedding_size": config["msa_averaging_embedding_size"],
            "num_msa_heads": config["num_msa_heads"],
            "msa_transition_multiplier": config["msa_transition_multiplier"],
            "num_triangular_update_channels": config["num_triangular_update_channels"],
            "num_triangular_attention_channels": config["num_triangular_attention_channels"],
            "num_triangular_attention_heads": config["num_triangular_attention_heads"],
            "num_template_blocks": config["num_template_blocks"],
            "template_embedding_size": config["template_embedding_size"],
            "num_pairformer_blocks": config["num_pairformer_blocks"],
            "num_pair_heads": config["num_pair_heads"],
            "pairformer_transition_multiplier": config["pairformer_transition_multiplier"],
            "diffusion_steps": config["diffusion_steps"],
            "diffusion_batch_size": config["diffusion_batch_size"],
            "atom_embedding_size": config["atom_embedding_size"],
            "atom_pair_embedding_size": config["atom_pair_embedding_size"],
            "token_embedding_size": config["token_embedding_size"],
            "num_diffusion_transformer_blocks": config["num_diffusion_transformer_blocks"],
            "num_diffusion_transformer_heads": config["num_diffusion_transformer_heads"],
            "fourier_embedding_size": config["fourier_embedding_size"],
            "num_distogram_bins": config["num_distogram_bins"],
        }

    @classmethod
    def from_config(cls, config):
        return cls(**cls.get_args(config))

    def run_pairformer(self, *args):
        if self.use_grad_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                lambda *inputs: self.pairformer(*inputs), *args, use_reentrant=False
            )
        return self.pairformer(*args)

    def get_total_loss(self, diffusion_loss, dist_loss):
        return 4 * diffusion_loss + 0.03 * dist_loss

    def forward(self, features):
        num_recycle = (
            torch.randint(self.num_recycle, (1,)) + 1 if self.training else self.num_recycle
        )

        input = self.input_embedder(features)
        single_rep_init = self.linear_input_single(input)
        pair_rep_init = self.linear_input_pair_a(input.unsqueeze(-2)) + self.linear_input_pair_b(
            input.unsqueeze(-3)
        )
        pair_rep_init = pair_rep_init + self.relative_position_encoding(features["residue_index"])
        single_rep_prev = torch.zeros_like(single_rep_init)
        pair_rep_prev = torch.zeros_like(pair_rep_init)

        for _ in range(num_recycle):
            pair_rep = pair_rep_init + self.transition_pair(pair_rep_prev)
            pair_rep = pair_rep + self.template_embedder(features, pair_rep)
            pair_rep = pair_rep + self.msa_module(features, pair_rep, input)
            single_rep = single_rep_init + self.transition_single(single_rep_prev)
            single_rep, pair_rep = self.run_pairformer(single_rep, pair_rep)
            single_rep_prev, pair_rep_prev = single_rep, pair_rep

        diffusion_loss = self.diffusion_model(features, input, single_rep, pair_rep)
        dist_loss = self.distogram_loss(pair_rep, features["translations"])
        return {
            "diffusion_loss": diffusion_loss,
            "dist_loss": dist_loss,
            "total_loss": self.get_total_loss(diffusion_loss, dist_loss),
        }

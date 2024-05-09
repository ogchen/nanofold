import torch
import torch.nn as nn


class AtomAttentionEncoder(nn.Module):
    def __init__(
        self,
        atom_embedding_size,
        atom_pair_embedding_size,
        token_embedding_size,
        num_block,
        num_head,
        num_queries,
        num_keys,
        atoms_per_residue=3,
    ):
        super().__init__()
        self.atoms_per_residue = atoms_per_residue
        self.linear_pos = nn.Linear(3, atom_embedding_size, bias=False)
        self.linear_pos_offset = nn.Linear(3, atom_pair_embedding_size, bias=False)
        self.linear_inv_sq_dist = nn.Linear(1, atom_pair_embedding_size, bias=False)
        self.linear_mask = nn.Linear(1, atom_pair_embedding_size, bias=False)
        self.conditioning_transition = nn.Sequential(
            nn.ReLU(), nn.Linear(atom_embedding_size, atom_pair_embedding_size, bias=False)
        )
        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_pair_embedding_size, atom_pair_embedding_size, bias=False),
            nn.ReLU(),
            nn.Linear(atom_pair_embedding_size, atom_pair_embedding_size, bias=False),
            nn.ReLU(),
            nn.Linear(atom_pair_embedding_size, atom_pair_embedding_size, bias=False),
        )
        self.atom_transformer = AtomTransformer(
            atom_embedding_size,
            atom_pair_embedding_size,
            num_block,
            num_head,
            num_queries,
            num_keys,
        )
        self.projection = nn.Sequential(
            nn.Linear(atom_embedding_size, token_embedding_size, bias=False),
            nn.ReLU(),
        )

    def forward(self, ref_pos, ref_space_uid, r, s, z):
        c = self.linear_pos(ref_pos)

        distance = ref_pos.unsqueeze(-2) - ref_pos.unsqueeze(-3)
        mask = (ref_space_uid.unsqueeze(-1) == ref_space_uid.unsqueeze(-2)).unsqueeze(-1)
        pair_rep = self.linear_pos_offset(distance) * mask
        squared_distance = (distance.unsqueeze(-2) @ distance.unsqueeze(-1)).squeeze(-1)
        pair_rep = pair_rep + self.linear_inv_sq_dist(1 / (1 + squared_distance)) * mask
        pair_rep = pair_rep + self.linear_mask(mask.float()) * mask

        q = c
        if r is not None:
            raise NotImplementedError("r is not None")
        pair_rep = (
            pair_rep
            + self.conditioning_transition(c.unsqueeze(-2))
            + self.conditioning_transition(c.unsqueeze(-3))
        )
        pair_rep = pair_rep + self.pair_mlp(pair_rep)
        q = self.projection(self.atom_transformer(q, c, pair_rep))
        a = torch.mean(q.view(-1, self.atoms_per_residue, q.size(-1)), dim=-2)
        return a, q, c, pair_rep

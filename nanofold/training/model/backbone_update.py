import torch
from torch import nn
from nanofold.training.frame import Frame


class BackboneUpdate(nn.Module):
    def __init__(self, single_embedding_size):
        super().__init__()
        self.linear = nn.Linear(single_embedding_size, 3 + 1 + 1 + 1)

    def forward(self, single):
        out = self.linear(single)
        bcd = out[..., :3]

        a = 1 / torch.sqrt(1 + bcd[..., 0] ** 2 + bcd[..., 1] ** 2 + bcd[..., 2] ** 2)
        quaternion = a.unsqueeze(-1) * bcd
        b, c, d = (quaternion[..., 0], quaternion[..., 1], quaternion[..., 2])

        r0 = torch.stack(
            [a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (a * c + b * d)], dim=-1
        )
        r1 = torch.stack(
            [2 * (b * c + a * d), a**2 - b**2 + c**2 - d**2, 2 * (c * d - a * b)], dim=-1
        )
        r2 = torch.stack(
            [2 * (b * d - a * c), 2 * (a * b + c * d), a**2 - b**2 - c**2 + d**2], dim=-1
        )
        rotations = torch.stack([r0, r1, r2], dim=-2)
        return Frame(rotations=rotations, translations=out[..., 3:])

import torch


def quaternion_to_rotation_matrix(quaternion):
    quaternion = torch.concat([torch.ones(quaternion.size(0), 1), quaternion], dim=-1)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)

    a, b, c, d = (quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3])

    r0 = torch.stack([a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (a * c + b * d)], dim=-1)
    r1 = torch.stack([2 * (b * c + a * d), a**2 - b**2 + c**2 - d**2, 2 * (c * d - a * b)], dim=-1)
    r2 = torch.stack([2 * (b * d - a * c), 2 * (a * b + c * d), a**2 - b**2 - c**2 + d**2], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


def uniform_random_rotation(*batch_size):
    quaternion = torch.rand(*batch_size, 3) * 2 if batch_size else torch.rand(1, 3) * 2
    rotation = quaternion_to_rotation_matrix(quaternion)
    return rotation if batch_size else rotation[0]

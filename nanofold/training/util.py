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


def rigid_align(x, x_truth):
    x_mean = x.mean(dim=-2, keepdim=True)
    x = x - x_mean
    x_truth = x_truth - x_truth.mean(dim=-2, keepdim=True)
    product = torch.einsum("...la,...lb->...ab", x_truth, x)
    U, _, V = torch.svd(product.float())
    R = U @ V
    is_reflection = torch.linalg.det(R.float()).unsqueeze(-1).unsqueeze(-1) < 0
    R = (
        R * ~is_reflection
        + (U @ torch.diag(torch.tensor([1.0, 1.0, -1.0], device=R.device)) @ V) * is_reflection
    )
    x_align = (R @ x.transpose(-2, -1)).transpose(-2, -1) + x_mean
    return x_align

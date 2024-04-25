import math
import torch

from nanofold.training.model import triangular_update


def test_triangle_multiplication_outgoing():
    batch = 2
    num_res = 16
    pair_embedding_size = 3
    num_channels = 4
    model = triangular_update.TriangleMultiplicationOutgoing(pair_embedding_size, num_channels)
    pair_rep = torch.rand([batch, num_res, num_res, pair_embedding_size])
    result = model(pair_rep)

    pair_rep = model.layer_norm_pair(pair_rep)
    for x in range(batch):
        for i in range(num_res):
            for j in range(num_res):
                sum = torch.zeros(num_channels)
                for k in range(num_res):
                    a = model.gate_a(pair_rep[x, i, k]) * model.linear_a(pair_rep[x, i, k])
                    b = model.gate_b(pair_rep[x, j, k]) * model.linear_b(pair_rep[x, j, k])
                    sum += a * b
                update = model.update_transition(sum)
                gate = model.gate(pair_rep[x, i, j])
                assert torch.allclose(result[x, i, j], gate * update, atol=1e-3)


def test_triangle_multiplication_incoming():
    batch = 2
    num_res = 16
    pair_embedding_size = 3
    num_channels = 4
    model = triangular_update.TriangleMultiplicationIncoming(pair_embedding_size, num_channels)
    pair_rep = torch.rand([batch, num_res, num_res, pair_embedding_size])
    result = model(pair_rep)

    pair_rep = model.layer_norm_pair(pair_rep)
    for x in range(batch):
        for i in range(num_res):
            for j in range(num_res):
                sum = torch.zeros(num_channels)
                for k in range(num_res):
                    a = model.gate_a(pair_rep[x, k, i]) * model.linear_a(pair_rep[x, k, i])
                    b = model.gate_b(pair_rep[x, k, j]) * model.linear_b(pair_rep[x, k, j])
                    sum += a * b
                update = model.update_transition(sum)
                gate = model.gate(pair_rep[x, i, j])
                assert torch.allclose(result[x, i, j], gate * update, atol=1e-3)

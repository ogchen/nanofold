import math
import torch

from nanofold.training.model import triangular_attention


def test_triangle_attention_starting_node():
    batch = 2
    num_heads = 2
    num_res = 16
    pair_embedding_size = 3
    num_channels = 4
    model = triangular_attention.TriangleAttentionStartingNode(
        pair_embedding_size, num_heads, num_channels
    )
    pair_rep = torch.rand([batch, num_res, num_res, pair_embedding_size])
    result = model(pair_rep)

    pair_rep = model.layer_norm(pair_rep)
    for x in range(batch):
        for i in range(num_res):
            for j in range(num_res):
                out = []
                for h in range(num_heads):
                    attn = []
                    for k in range(num_res):
                        query = model.query(pair_rep[x, i, j])[h]
                        key = model.key(pair_rep[x, i, k])[h]
                        bias = model.bias(pair_rep[x, j, k])[h]
                        attn.append(query.T @ key / math.sqrt(num_channels) + bias)
                    attn = torch.nn.functional.softmax(torch.stack(attn))
                    sum = 0
                    for k in range(num_res):
                        value = model.value(pair_rep[x, i, k])[h]
                        sum += attn[k] * value
                    gate = model.gate(pair_rep[x, i, j])[h]
                    out.append(gate * sum)
                out = torch.stack(out).reshape(-1)
                torch.allclose(result[x, i, j], model.out_proj(out), atol=1e-3)

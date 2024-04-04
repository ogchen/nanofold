import math
import torch

from nanofold.training.model import msa_attention


def test_msa_row_attention_with_pair_bias():
    batch = 2
    num_heads = 2
    num_res = 16
    pair_embedding_size = 3
    msa_embedding_size = 4
    num_msa = 5
    model = msa_attention.MSARowAttentionWithPairBias(
        pair_embedding_size, msa_embedding_size, num_heads, num_channels=6
    )
    pair_rep = torch.rand([batch, num_res, num_res, pair_embedding_size])
    msa_rep = torch.rand([batch, num_msa, num_res, msa_embedding_size])
    result = model(msa_rep, pair_rep)

    for b in range(batch):
        for s in range(num_msa):
            m = model.layer_norm(msa_rep[b, s])
            for i in range(num_res):
                o = []
                for h in range(num_heads):
                    q = model.query(m[i])[h]
                    a = []
                    for j in range(num_res):
                        k = model.key(m[j])[h]
                        bias = model.bias(pair_rep[b, i, j])[h]
                        a.append((q.T @ k) / math.sqrt(model.num_channels) + bias)

                    a = torch.nn.functional.softmax(torch.stack(a))
                    sum = 0
                    for j in range(num_res):
                        sum += a[j] * model.value(m[j])[h]
                    g = model.gate(m[i])[h]
                    o.append(g * sum)
                o = torch.stack(o).reshape(-1)
                assert torch.allclose(result[b][s][i], model.projection(o), atol=1e-3)


def test_msa_col_attention():
    batch = 2
    num_heads = 2
    num_res = 16
    msa_embedding_size = 4
    num_msa = 5
    model = msa_attention.MSAColumnAttention(msa_embedding_size, num_heads, num_channels=6)
    msa_rep = torch.rand([batch, num_msa, num_res, msa_embedding_size])
    result = model(msa_rep)

    msa_rep = model.layer_norm(msa_rep)
    for b in range(batch):
        for s in range(num_msa):
            for i in range(num_res):
                o = []
                for h in range(num_heads):
                    a = []
                    for t in range(num_msa):
                        q = model.query(msa_rep[b, s, i])[h]
                        k = model.key(msa_rep[b, t, i])[h]
                        a.append((q.T @ k) / math.sqrt(model.num_channels))
                    a = torch.nn.functional.softmax(torch.stack(a))
                    sum = 0
                    for t in range(num_msa):
                        sum += a[t] * model.value(msa_rep[b, t, i])[h]
                    g = model.gate(msa_rep[b, s, i])[h]
                    o.append(g * sum)
                o = torch.stack(o).reshape(-1)
                assert torch.allclose(result[b][s][i], model.projection(o), atol=1e-3)

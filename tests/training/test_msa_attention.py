import math
import torch

from nanofold.training.model.msa_attention import MSARowAttentionWithPairBias


def test_msa_row_attention_with_pair_bias():
    batch = 2
    num_heads = 2
    num_res = 16
    pair_embedding_size = 3
    msa_embedding_size = 4
    num_msa = 5
    model = MSARowAttentionWithPairBias(
        pair_embedding_size, msa_embedding_size, num_heads, num_channels=6
    )
    pair_rep = torch.rand([batch, num_res, num_res, pair_embedding_size])
    msa_rep = torch.rand([batch, num_msa, num_res, msa_embedding_size])
    result = model(msa_rep, pair_rep)  #

    for b in range(batch):
        for s in range(num_msa):
            m = model.layer_norm_msa(msa_rep[b, s])
            for i in range(num_res):
                o = []
                for h in range(num_heads):
                    q = model.query(m[i])[h]
                    a = []
                    for j in range(num_res):
                        k = model.key(m[j])[h]
                        bias = model.linear_pair(model.layer_norm_pair(pair_rep[b, i, j]))[h]
                        a.append((q.T @ k) / math.sqrt(model.num_channels) + bias)

                    a = torch.nn.functional.softmax(torch.stack(a))
                    sum = 0
                    for j in range(num_res):
                        sum += a[j] * model.value(m[j])[h]
                    g = torch.sigmoid(model.linear_msa(m[i])[h])
                    o.append(g * sum)
                o = torch.stack(o).reshape(-1)
                assert torch.allclose(result[b][s][i], model.projection(o), atol=1e-3)

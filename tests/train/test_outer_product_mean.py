import torch

from nanofold.train.model import outer_product_mean


def test_outer_product_mean():
    batch = 2
    num_msa = 4
    num_res = 16
    pair_embedding_size = 3
    msa_embedding_size = 4
    product_embedding_size = 5
    model = outer_product_mean.OuterProductMean(
        pair_embedding_size, msa_embedding_size, product_embedding_size
    )
    msa_rep = torch.rand([batch, num_msa, num_res, msa_embedding_size])
    result = model(msa_rep)

    msa_rep = model.layer_norm(msa_rep)
    for x in range(batch):
        for i in range(num_res):
            for j in range(num_res):
                o = []
                for c1 in range(product_embedding_size):
                    for c2 in range(product_embedding_size):
                        prod = []
                        for s in range(num_msa):
                            a = model.linear_a(msa_rep[x, s, i])[c1]
                            b = model.linear_b(msa_rep[x, s, j])[c2]
                            prod.append(a * b)
                        o.append(torch.stack(prod).mean())
                o = torch.stack(o)
                assert torch.allclose(result[x, i, j], model.projection(o), atol=1e-3)

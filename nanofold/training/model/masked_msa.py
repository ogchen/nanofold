from torch import nn

from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA_WITH_MASK


class MaskedMSAPredictor(nn.Module):
    def __init__(self, msa_embedding_size):
        super().__init__()
        self.linear = nn.Linear(msa_embedding_size, len(RESIDUE_INDEX_MSA_WITH_MASK))

    def forward(self, msa_rep, msa_mask, msa_truth):
        logits = self.linear(msa_rep)
        loss = msa_mask * nn.functional.cross_entropy(
            logits.transpose(1, -1),
            msa_truth.transpose(1, -1),
            reduction="none",
        ).transpose(1, -1)
        return loss.sum() / msa_mask.sum()

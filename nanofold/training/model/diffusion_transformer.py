class DiffusionTransformer(nn.Module):
    def __init__(self, atom_embedding_size, atom_pair_embedding_size, num_block, num_head):
        super().__init__()
        self.num_block = num_block
        self.num_head = num_head
        self.attention_pair_bias = AttentionPairBias(
            num_head, atom_embedding_size, atom_embedding_size, atom_pair_embedding_size
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            atom_embedding_size, atom_embedding_size
        )

    def forward(self, a, s, pair_rep, beta):
        for _ in range(self.num_block):
            b = self.attention_pair_bias(a, s, pair_rep, beta)
            a = b + self.conditioned_transition_block(a, s)
        return a

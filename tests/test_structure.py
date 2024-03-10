import torch
from nanofold.frame import Frame
from nanofold.model.structure import StructureModuleLayer
class TestStructureModuleLayer:
    def setup_method(self):
        self.single_embedding_size = 4
        self.len_seq = 7
        self.pair_embedding_size = 5
        self.model = StructureModuleLayer(
            single_embedding_size=self.single_embedding_size,
            pair_embedding_size=self.pair_embedding_size,
            ipa_embedding_size=6,
            num_query_points=3,
            num_value_points=3,
            num_heads=2,
            dropout=0.1,
        )
        self.len_seq = 7
        self.single_embedding_size = 4
        self.pair_embedding_size = 5
        self.single = torch.rand(self.len_seq, self.single_embedding_size)
        self.pair = torch.rand(self.len_seq, self.len_seq, self.pair_embedding_size)
        self.frames = Frame(
            rotations=torch.eye(3).unsqueeze(0).repeat(self.len_seq, 1, 1),
            translations=torch.zeros(self.len_seq, 3),
        )

    def test_structure_module_layer(self):
        s, f, loss = self.model(self.single, self.pair, self.frames)
        assert s.shape == self.single.shape
        assert f.rotations.shape == self.frames.rotations.shape
        assert f.translations.shape == self.frames.translations.shape
        assert torch.allclose(f.rotations @ f.rotations.transpose(-2, -1), torch.eye(3), atol=1e-5)
        assert loss is None

    def test_structure_module_layer_loss(self):
        s, f, loss = self.model(self.single, self.pair, self.frames, self.frames)
        assert loss is not None
        # Check no exception raised when we traverse the graph
        loss.backward()

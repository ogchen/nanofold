import torch
from nanofold.training.frame import Frame
from nanofold.training.model.structure import StructureModuleLayer
from nanofold.training.model.structure import StructureModule


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

    def test_structure_module_layer_batched(self):
        self.model.eval()  # Ensure dropout is not applied
        s, f, loss = self.model(self.single, self.pair, self.frames, self.frames)
        batch_frames = Frame(
            rotations=torch.stack([self.frames.rotations] * 2),
            translations=torch.stack([self.frames.translations] * 2),
        )
        batch_s, batch_f, batch_loss = self.model(
            torch.stack([self.single] * 2), torch.stack([self.pair] * 2), batch_frames, batch_frames
        )
        for i in range(2):
            assert torch.allclose(s, batch_s[i], atol=1e-3)
            assert torch.allclose(f.rotations, batch_f.rotations[i], atol=1e-3)
            assert torch.allclose(f.translations, batch_f.translations[i], atol=1e-3)
        assert torch.allclose(loss, batch_loss, atol=1e-3)


class TestStructureModule:
    def setup_method(self):
        self.single_embedding_size = 4
        self.len_seq = 7
        self.pair_embedding_size = 5
        self.model = StructureModule(
            num_layers=3,
            single_embedding_size=self.single_embedding_size,
            pair_embedding_size=self.pair_embedding_size,
            ipa_embedding_size=6,
            num_query_points=3,
            num_value_points=3,
            num_heads=2,
            dropout=0.1,
        )
        self.len_seq = 5
        self.sequence = ["MET", "PHE", "PRO", "SER", "THR"]
        self.local_coords = torch.zeros(self.len_seq, 3, 3)
        self.single_embedding_size = 4
        self.pair_embedding_size = 5
        self.single = torch.rand(self.len_seq, self.single_embedding_size)
        self.pair = torch.rand(self.len_seq, self.len_seq, self.pair_embedding_size)
        self.frames_truth = Frame(
            rotations=torch.eye(3).unsqueeze(0).repeat(self.len_seq, 1, 1),
            translations=torch.zeros(self.len_seq, 3),
        )

    def test_structure_module(self):
        coords, _, _ = self.model(self.single, self.pair, self.local_coords)
        assert coords.shape == (self.len_seq, 3, 3)

    def test_structure_module_loss(self):
        _, aux_loss, fape_loss = self.model(
            self.single, self.pair, self.local_coords, self.frames_truth
        )
        assert aux_loss is not None
        assert fape_loss is not None
        # Check no exception raised when we traverse the graph
        (aux_loss + fape_loss).backward()

    def test_structure_module_batched(self):
        self.model.eval()  # Ensure dropout is not applied
        coords, aux_loss, fape_loss = self.model(
            self.single, self.pair, self.local_coords, self.frames_truth
        )
        batch_coords, batch_aux_loss, batch_fape_loss = self.model(
            torch.stack([self.single] * 2),
            torch.stack([self.pair] * 2),
            torch.stack([self.local_coords] * 2),
            Frame(
                rotations=torch.stack([self.frames_truth.rotations] * 2),
                translations=torch.stack([self.frames_truth.translations] * 2),
            ),
        )
        assert torch.allclose(coords, batch_coords[0], atol=1e-3)
        assert torch.allclose(coords, batch_coords[1], atol=1e-3)
        assert torch.allclose(aux_loss, batch_aux_loss, atol=1e-3)
        assert torch.allclose(fape_loss, batch_fape_loss, atol=1e-3)

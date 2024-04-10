import math
import torch
from nanofold.training.frame import Frame
from nanofold.training.loss import compute_fape_loss


class TestFapeLoss:
    def setup_method(self):
        self.frames = Frame(
            rotations=torch.stack(
                [
                    torch.eye(3),
                    self.create_rotation(1),
                ]
            ),
            translations=torch.stack([torch.zeros(3), torch.tensor([20, 10, 10])]),
        )
        self.coords = torch.stack(
            [torch.zeros(3), torch.tensor([10, 20, 30]), torch.tensor([1, 2, 3])]
        )
        self.frames_truth = Frame(
            torch.stack([torch.eye(3), torch.eye(3)]),
            torch.stack([torch.zeros(3), torch.zeros(3)]),
        )
        self.coords_truth = 10 * torch.ones(len(self.coords), 3)

    def create_rotation(self, angle):
        cos = math.cos(angle)
        sin = math.sin(angle)
        return torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])

    def test_loss_fape(self):
        length_scale = 10
        clamp = 20
        loss = compute_fape_loss(
            self.frames,
            self.coords,
            self.frames_truth,
            self.coords_truth,
            length_scale=length_scale,
            clamp=clamp,
        )

        expected = 0
        for i in range(len(self.frames)):
            for j in range(len(self.coords)):
                inverse = Frame.inverse(self.frames[i])
                inverse_truth = Frame.inverse(self.frames_truth[i])
                glob_coord = Frame.apply(inverse, self.coords[j])
                glob_coord_truth = Frame.apply(inverse_truth, self.coords_truth[j])
                expected += min(clamp, abs(glob_coord - glob_coord_truth).norm())
        expected /= len(self.frames) * len(self.coords) * length_scale
        assert torch.isclose(loss, expected)

        batched_frames = Frame(
            torch.stack([self.frames.rotations, self.frames.rotations]),
            torch.stack([self.frames.translations, self.frames.translations]),
        )
        batched_frames_truth = Frame(
            torch.stack([self.frames_truth.rotations, self.frames_truth.rotations]),
            torch.stack([self.frames_truth.translations, self.frames_truth.translations]),
        )

        batched = compute_fape_loss(
            batched_frames,
            torch.stack([self.coords, self.coords]),
            batched_frames_truth,
            torch.stack([self.coords_truth, self.coords_truth]),
            length_scale=length_scale,
            clamp=clamp,
        )

        assert torch.allclose(loss, batched, atol=1e-3)

    def test_loss_fape_invariance(self):
        loss = compute_fape_loss(
            self.frames,
            self.coords,
            self.frames_truth,
            self.coords_truth,
            clamp=None,
        )
        assert loss > 0
        assert torch.isclose(
            loss,
            compute_fape_loss(
                self.frames_truth,
                self.coords_truth,
                self.frames,
                self.coords,
                clamp=None,
            ),
            atol=1e-3,
        )
        assert torch.isclose(
            torch.zeros(1),
            compute_fape_loss(
                self.frames,
                self.coords,
                self.frames,
                self.coords,
                clamp=None,
            ),
            atol=1e-3,
        )

        frame_offset = Frame(
            self.create_rotation(0.5),
            torch.tensor([10, 10, 10]),
        )
        assert torch.isclose(
            loss,
            compute_fape_loss(
                Frame.compose(frame_offset, self.frames),
                Frame.apply(frame_offset, self.coords),
                self.frames_truth,
                self.coords_truth,
                clamp=None,
            ),
            atol=1e-3,
        )

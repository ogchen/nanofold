from nanofold.frame import Frame
import pytest
import torch
import math


def test_frame_init():
    with pytest.raises(ValueError):
        Frame(torch.stack([torch.eye(3), torch.eye(3)]), torch.stack([torch.zeros(3)]))
    with pytest.raises(ValueError):
        Frame(torch.stack([torch.ones(3)]), torch.stack([torch.zeros(3)]))
    Frame(torch.stack([torch.eye(3)]), torch.stack([torch.zeros(3)]))


def test_frame_inverse():
    cos = math.cos(1)
    sin = math.sin(1)
    rotations = torch.stack(
        [
            torch.eye(3),
            torch.tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]),
            torch.tensor([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]),
        ]
    )
    translations = torch.stack([torch.zeros(3), torch.ones(3), torch.tensor([1, 2, 3])])

    frames = Frame(rotations, translations)
    inverse = Frame.inverse(frames)
    x = torch.tensor([1, 2, 3]).float()
    result = Frame.apply(inverse, Frame.apply(frames, x))
    assert torch.allclose(result, x)


def test_frame_compose():
    rotationsA = torch.stack(
        [
            torch.eye(3),
            torch.diag(torch.tensor([2, 0.5, 1])),
            torch.diag(torch.tensor([4, 2, 2])),
        ]
    )
    rotationsB = torch.stack(
        [
            torch.eye(3),
            torch.eye(3),
            torch.diag(torch.tensor([0.5, 0.5, 0.5])),
        ]
    )
    translationsA = torch.stack(
        [torch.zeros(3), torch.ones(3), torch.tensor([1, 2, 3])]
    )
    translationsB = torch.stack([torch.zeros(3), torch.ones(3), -torch.ones(3)])

    expected_rotations = torch.stack(
        [
            torch.eye(3),
            torch.diag(torch.tensor([2, 0.5, 1])),
            torch.diag(torch.tensor([2, 1, 1])),
        ]
    )
    expected_translations = torch.stack(
        [torch.zeros(3), torch.tensor([3, 1.5, 2]), torch.tensor([-3, 0, 1])]
    )
    expected = Frame(expected_rotations, expected_translations)
    framesA = Frame(rotationsA, translationsA)
    framesB = Frame(rotationsB, translationsB)
    res = Frame.compose(framesA, framesB)
    assert torch.allclose(
        res.rotations, expected.rotations
    ), f"{res.rotations} != {expected.rotations}"
    assert torch.allclose(
        res.translations, expected.translations
    ), f"{res.translations} != {expected.translations}"


def test_frame_apply():
    rotations = torch.stack(
        [
            torch.eye(3),
            2 * torch.eye(3),
            torch.ones(3, 3),
        ]
    )
    translations = torch.stack([torch.zeros(3), torch.ones(3), torch.tensor([1, 2, 3])])
    frames = Frame(rotations.unsqueeze(1), translations.unsqueeze(1))
    vectors = torch.stack([torch.ones(3), -torch.ones(3), torch.tensor([1, 0, -1])])
    expected = torch.stack(
        [
            vectors,
            torch.stack([3 * torch.ones(3), -torch.ones(3), torch.tensor([3, 1, -1])]),
            torch.tensor([[4, 5, 6], [-2, -1, 0], [1, 2, 3]]),
        ]
    )
    res = Frame.apply(frames, vectors)
    assert torch.allclose(res, expected), f"{res} != {expected}"


def test_frame_add():
    frames1 = Frame(
        torch.stack([torch.eye(3), 2 * torch.eye(3)]),
        torch.stack([torch.ones(3), 2 * torch.ones(3)]),
    )
    frames2 = Frame(
        torch.stack([3 * torch.eye(3), 4 * torch.eye(3)]),
        torch.stack([3 * torch.ones(3), 4 * torch.ones(3)]),
    )
    expected = Frame(
        torch.stack(
            [torch.eye(3), 2 * torch.eye(3), 3 * torch.eye(3), 4 * torch.eye(3)]
        ),
        torch.stack(
            [torch.ones(3), 2 * torch.ones(3), 3 * torch.ones(3), 4 * torch.ones(3)]
        ),
    )
    result = frames1 + frames2
    assert torch.allclose(result.rotations, expected.rotations)
    assert torch.allclose(result.translations, expected.translations)


def test_frame_getitem():
    frames = Frame(
        torch.stack([torch.eye(3), 2 * torch.eye(3), 3 * torch.eye(3)]),
        torch.stack([torch.ones(3), 2 * torch.ones(3), 3 * torch.ones(3)]),
    )
    expected = Frame(
        torch.stack([2 * torch.eye(3)]),
        torch.stack([2 * torch.ones(3)]),
    )
    result = frames[1]
    assert torch.allclose(result.rotations, expected.rotations)
    assert torch.allclose(result.translations, expected.translations)


def test_frame_getitem_slice():
    frames = Frame(
        torch.stack([torch.eye(3), 2 * torch.eye(3), 3 * torch.eye(3)]),
        torch.stack([torch.ones(3), 2 * torch.ones(3), 3 * torch.ones(3)]),
    )
    expected = Frame(
        torch.stack([2 * torch.eye(3), 3 * torch.eye(3)]),
        torch.stack([2 * torch.ones(3), 3 * torch.ones(3)]),
    )
    result = frames[1:]
    assert torch.allclose(result.rotations, expected.rotations)
    assert torch.allclose(result.translations, expected.translations)

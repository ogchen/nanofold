import torch


class Frame:
    def __init__(self, rotations, translations):
        i, a, b = rotations.shape
        j, c = translations.shape
        assert i == j
        assert (a, b, c) == (3, 3, 3)
        self.rotations = rotations
        self.translations = translations

    @staticmethod
    def inverse(frame):
        inverse_rotations = torch.inverse(frame.rotations)
        inverse_translations = -inverse_rotations @ frame.translations.unsqueeze(-1)
        return Frame(inverse_rotations, inverse_translations.squeeze(-1))

    @staticmethod
    def allclose(a, b):
        return torch.allclose(a.rotations, b.rotations) and torch.allclose(
            a.translations, b.translations
        )

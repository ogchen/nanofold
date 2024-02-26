import torch


class Frame:
    def __init__(self, rotations, translations):
        i, a, b = rotations.shape
        j, c = translations.shape
        if i != j or (a, b, c) != (3, 3, 3):
            raise ValueError(
                f"Expected rotations to have shape (n, 3, 3) and translations to have shape (n, 3), got {rotations.shape} and {translations.shape}"
            )
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

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

    def __repr__(self):
        return f"Frame(rotations={self.rotations},\n translations={self.translations})"

    @staticmethod
    def inverse(frame):
        inverse_rotations = torch.inverse(frame.rotations)
        inverse_translations = -inverse_rotations @ frame.translations.unsqueeze(-1)
        return Frame(inverse_rotations, inverse_translations.squeeze(-1))

    @staticmethod
    def compose(a, b):
        rotations = a.rotations @ b.rotations
        translations = a.rotations @ b.translations.unsqueeze(
            -1
        ) + a.translations.unsqueeze(-1)
        return Frame(rotations, translations.squeeze(-1))

    @staticmethod
    def apply(frames, vectors):
        result = vectors @ frames.rotations.transpose(-2, -1)
        result = result.transpose(-2, -3) + frames.translations
        return result.transpose(-2, -3)

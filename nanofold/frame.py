import torch


class Frame:
    def __init__(self, rotations=torch.empty(0, 3, 3), translations=torch.empty(0, 3)):
        if len(rotations.shape) == 2 and len(translations.shape) == 1:
            rotations = rotations.unsqueeze(0)
            translations = translations.unsqueeze(0)
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

    def __add__(self, other):
        rotations = torch.cat([self.rotations, other.rotations])
        translations = torch.cat([self.translations, other.translations])
        return Frame(rotations, translations)

    @staticmethod
    def inverse(frame):
        inverse_rotations = torch.inverse(frame.rotations)
        inverse_translations = -inverse_rotations @ frame.translations.unsqueeze(-1)
        return Frame(inverse_rotations, inverse_translations.squeeze(-1))

    @staticmethod
    def compose(a, b):
        if a.rotations.shape[0] != b.rotations.shape[0]:
            raise ValueError(
                f"Expected a and b to have the same number of frames, got {a.rotations.shape[0]} and {b.rotations.shape[0]}"
            )
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

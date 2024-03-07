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

    def __len__(self):
        return len(self.rotations)

    def __add__(self, other):
        rotations = torch.cat([self.rotations, other.rotations])
        translations = torch.cat([self.translations, other.translations])
        return Frame(rotations, translations)

    def __getitem__(self, key):
        rotations = self.rotations[key]
        translations = self.translations[key]
        return Frame(rotations, translations)

    @staticmethod
    def inverse(frame):
        inverse_rotations = frame.rotations.transpose(-2, -1)
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
        result = frames.rotations @ vectors.unsqueeze(-1)
        result = result.squeeze(-1) + frames.translations
        return result

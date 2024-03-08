import torch


class Frame:
    def __init__(self, rotations=torch.empty(0, 3, 3), translations=torch.empty(0, 3)):
        r = rotations.shape
        t = translations.shape
        if len(r) == 2 and len(t) == 1:
            rotations = rotations.unsqueeze(0)
            translations = translations.unsqueeze(0)
        if r[:-2] != t[:-1] or (r[-2], r[-1], t[-1]) != (3, 3, 3):
            raise ValueError(
                f"Expected rotations to have shape (3, 3) and translations to have shape (3) with equal batch dimensions, got {rotations.shape} and {translations.shape}"
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
        translations = Frame.apply(a, b.translations)
        return Frame(rotations, translations)

    @staticmethod
    def apply(frames, vectors):
        result = frames.rotations @ vectors.unsqueeze(
            -1
        ) + frames.translations.unsqueeze(-1)
        return result.squeeze(-1)

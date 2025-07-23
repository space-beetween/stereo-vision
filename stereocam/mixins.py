import numpy as np


class NpzMixin:
    def save(self, filename: str) -> None:
        np.savez_compressed(filename, **self.__dict__)

    @classmethod
    def load(cls, filename: str):
        data = np.load(filename)
        return cls(
            **{
                k: (float(data[k]) if k == "reprojection_error" else data[k])
                for k in data.files
            }
        )

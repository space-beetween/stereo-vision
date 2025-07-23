import abc
from collections.abc import Iterator
from typing import Union, Sequence
from pathlib import Path


class AbstractDataset(abc.ABC):
    def __init__(
        self,
        path: Union[str, Path]
    ) -> None:
        if isinstance(path, str):
            self.folder = Path(path)
        else:
            self.folder = path

        self._load()

    @abc.abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Sequence:
        pass

    @abc.abstractmethod
    def _load(self):
        pass

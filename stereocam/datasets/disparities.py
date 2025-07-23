from typing import Union, List, Tuple, Sequence
from collections.abc import Iterator

import pathlib
import cv2
import numpy as np

from .frames import FramesDataset
from ..abc import AbstractDataset
from ..types import MapImagesPair
from ..calibration import DisparityMap


class DisparityDataset(AbstractDataset):
    def __init__(
        self,
        path: Union[str, pathlib.Path]
    ) -> None:
        super().__init__(path)

    def __len__(self):
        return len(self.disparities)

    def __iter__(self) -> "DisparityIterator":
        return DisparityIterator(self.frames, self.disparities)

    def __getitem__(self, index) -> Sequence:
        if index < 0 or index >= len(self):
            raise IndexError("Индекс вне диапазона")
        return self.frames[index], self.disparities[index].disparity

    def _load(self):
        self.frames_folder = self.folder / "frames"
        self.frames = FramesDataset(self.frames_folder)
        file_paths = sorted(self.folder.glob("*.npz"), key=lambda f: f.stat().st_mtime)
        self.disparities = [DisparityMap.load(disparity) for disparity in file_paths]


class DisparityIterator(Iterator):
    def __init__(
        self,
        frames: FramesDataset,
        disparites: List[DisparityMap]
    ) -> None:
        self.frames = frames
        self.disparities = disparites
        self.pos = 0

    def __next__(self) -> MapImagesPair:
        if self.pos >= len(self.frames):
            raise StopIteration
        frame_pair = self.frames[self.pos]
        disparity_map = self.disparities[self.pos].disparity
        self.pos += 1
        return frame_pair, disparity_map

    def __iter__(self):
        return self

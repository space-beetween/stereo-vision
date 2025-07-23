from typing import Union, List, Sequence
from collections.abc import Iterator

import pathlib
import cv2

from ..stereo_pair import StereoPair
from ..abc import AbstractDataset
from ..types import ImagePair


class FramesDataset(AbstractDataset):
    def __init__(
        self,
        path: Union[str, pathlib.Path]
    ) -> None:
        """
        Класс для работы с датасетом

        Параметры
        ---------
        path: :class:`str` | :class:`Path`
            Путь к папке с датасетом

        Аттрибуты
        ---------
        folder: :class:`Path`
            Путь к папке с датасетом
        """
        super().__init__(path)

    def __iter__(self) -> "FramesIterator":
        return FramesIterator(self.images)

    def __len__(self) -> int:
        return len(self.images) // 2

    def __getitem__(self, index: int) -> Sequence:
        if index < 0 or index >= len(self):
            raise IndexError("Индекс вне диапазона")
        return self.images[index * 2:index * 2 + 2]

    def _load(self) -> None:
        file_paths = sorted(self.folder.glob("*"), key=lambda f: f.stat().st_mtime)
        self.images = [cv2.imread(str(image)) for image in file_paths]
        images_amount = len(self.images)
        amount_valid = (images_amount != 0 and images_amount % 2 == 0)
        if not amount_valid:
            raise ValueError(f"Неккоректное количество изображений {images_amount}")

    def show(self) -> None:
        """Показывает все изображения внутри датасета
        Для переключения изображений достаточно нажать кнопку на клавиатуре
        """
        for frames in FramesDataset("dataset"):
            for idx, frame in enumerate(frames):
                cv2.imshow(StereoPair._sides[idx], frame)
            cv2.waitKey(0)


class FramesIterator(Iterator):
    def __init__(
        self,
        images: ImagePair
    ) -> None:
        self.images = images
        self.pos = 0

    def __next__(self) -> ImagePair:
        if self.pos + 1 >= len(self.images):
            raise StopIteration
        pair = self.images[self.pos:self.pos + 2]
        self.pos += 2
        return pair

    def __iter__(self) -> "FramesIterator":
        return self

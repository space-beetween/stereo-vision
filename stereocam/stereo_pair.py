from typing import List, Optional, Tuple

import cv2
import numpy as np

from .disparity_estimator import DisparityEstimator
from .types import ImagePair


class StereoPair:
    """Класс для работы с видеопотоком двух вебкамер

    Параметры
    ---------
    device_ids: List[:class:`int`]
            Список индексов камер, которые будут источником видеопотока.
            Нулевому индексу должна соответствовать левая камера, первому - правая.

    Аттрибуты
    ---------
    captures: List[:class:`VideoCapture`]
        Список обьектов VideoCapture, для чтения видеопотока.
        Нулевому индексу соответствует левая камера, первому - правая

    """

    _sides = ["left", "right"]

    def __init__(self, device_ids: List[int]) -> None:
        self._saved_frames_count = 0
        self._windows = [side for side in self._sides]
        self.captures = [cv2.VideoCapture(device_id, cv2.CAP_DSHOW) for device_id in device_ids]

    def get_frames(self) -> ImagePair:
        """
        Считывает кадры с камер

        Вовзращает
        ----------
        :class:`ImagePair`
            Список считанных кадров с левой и правой камеры
        """
        return [capture.read()[1] for capture in self.captures]

    def show_frames(self) -> None:
        """
        Показывает кадры с камер
        """
        for window, frame in zip(self._windows, self.get_frames()):
            cv2.imshow(window, frame)

    def show_videos(self) -> None:
        """
        Транслирует видеопоток с камер

        Для закрытия окон достаточно нажать q
        """
        while True:
            self.show_frames()
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == ord('q'):
                break

    def save_frames(
        self,
        fp: str,
        frames: Optional[ImagePair] = None
    ) -> None:
        """
        Записывает кадры с камер в хранилище

        Параметры
        ---------
        fp: :class:`str`
            Путь к папке, в которую записываются кадры

        frames: Optional[List[:class:`ImagePair`]]
            Список кадров с камер, которые необходимо записать
            Если не указано - захватываются текущие кадры
        """
        if frames is None:
            frames = self.get_frames()
        for idx, frame in enumerate(frames):
            side = self._sides[idx]
            cv2.imwrite(f'{fp}/{self._saved_frames_count}_{side}.png', frame)
        self._saved_frames_count += 1

    def get_chessboard_frames(
            self,
            columns: int,
            rows: int,
    ) -> Tuple[ImagePair]:
        """
        Получает кадры с камер с шахматной доской

        Этот метод дожидается пока на обоих кадрах не будут найдены углы шахматной доски.

        Параметры
        ---------
        columns :class:`int`
            Количество внутренних углов в столбцах шахматной доски

        rows: :class:`int`
            Количество внутренних углов в строках шахматной доски

        Возвращает
        ----------
        frames: :class:`ImagesPair`
            Список кадров с камер
        chessboard_corners List[:class:`MatLike`]
            Найденные углы.
        """
        found = [False, False]
        chessboard_corners = [None, None]

        while not all(found):
            print("Поиск доски...")
            frames = self.get_frames()
            for idx, frame in enumerate(frames):
                found[idx], chessboard_corners[idx] = cv2.findChessboardCorners(
                    frame,
                    (columns, rows),
                    flags=cv2.CALIB_CB_FAST_CHECK
                )

        return frames, chessboard_corners

    def show_disparity_map(
        self,
        disparity_estimator: DisparityEstimator
    ) -> None:
        """
        Строит и показывает карту диспаратности
        Для выхода из метода нужно нажать q

        Параметры
        ---------
        disparity_estimator: :class:`DisparityEstimator`
            Экземпляр класса DisparityEstimator, который высчитывает карту диспаратности
        """
        while True:
            frames = self.get_frames()
            disparity = disparity_estimator.compute(frames)
            visualization = cv2.normalize(
                disparity,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )
            cv2.imshow("Disparity", visualization)
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == ord('q'):
                break

    def save_disparity(
        self,
        fp: str,
        disparity_estimator: DisparityEstimator,
        frames: Optional[ImagePair] = None
    ) -> None:
        """
        Сохраняет карту диспаратности в файл

        Параметры
        ----------
        fp: :class:`str`
            Путь к файлу, в который будет сохранена карта диспаратности

        disparity_estimator: :class:`DisparityEstimator`
            Экземпляр класса DisparityEstimator, который высчитывает карту диспаратности

        frames: Optional[:class:`ImagePair`]
            Список кадров с камер, которые необходимо использовать
            для вычисления карты диспаратности

            Если не указано - захватываются текущие кадры
        """
        disparity_estimator.set_mode(cv2.STEREO_SGBM_MODE_HH)
        if frames is None:
            frames = self.get_frames()
        disparity = disparity_estimator.get_filtered_disparity(frames)
        np.savez_compressed(fp, disparity=disparity)

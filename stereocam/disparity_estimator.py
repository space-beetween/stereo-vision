from typing import List

import cv2
import numpy as np

from .calibration import TransformationMap
from .sgbm_config import SGBMConfig
from .types import ImagePair


class DisparityEstimator:
    """
    Класс для расчета карты диспаратности

    Параметры
    ---------
    transformation: :class:`TransformationMap`
        Датакласс, содержащий в себе массивы для ремаппинга

    Аттрибуты
    ---------
    matcher: :class:`StereoMatcher`
        Реализация алгоритма для сопоставления изображений
    mode: :class:`int`
        Режим алгоритма SGBM
    """
    def __init__(
        self,
        tranformation: TransformationMap
    ) -> None:
        self.transformation = tranformation
        self.mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        self.matcher = SGBMConfig.from_path('sgbm_config.yml').get_matcher(self.mode)

    def compute(
        self,
        frames: ImagePair
    ) -> np.ndarray:
        """
        Вычисляет карту диспаратности

        Параметры
        ---------
        frames: :class:`Image`
            Левый и правый кадр

        Возвращает
        ----------
        :class:`ndarray`
            Карта диспаратности
        """
        # трансформируем левый и правый кадр
        frames[0] = cv2.remap(
            frames[0],
            self.transformation.left_undistortion_map,
            self.transformation.left_rectification_map,
            cv2.INTER_LINEAR
        )

        frames[1] = cv2.remap(
            frames[1],
            self.transformation.right_undistortion_map,
            self.transformation.right_rectification_map,
            cv2.INTER_LINEAR
        )

        for idx, frame in enumerate(frames):
            frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        disparity = self.matcher.compute(*frames)

        return disparity.astype(np.float32) / 16.0

    def set_mode(self, mode: int) -> None:
        """Меняет режим алгоритма"""
        if mode == self.mode:
            return
        self.mode = mode
        self.matcher = SGBMConfig.from_path('sgbm_config.yml').get_matcher(self.mode)

    def get_filtered_disparity(
        self,
        frames: ImagePair,
    ) -> np.ndarray:
        """
        Вычисляет карту диспаратности и применяет WSL фильтрацию

        Параметры
        ---------
        frames: :class:`Image`
            Левый и правый кадр

        Возвращает
        ----------
        :class:`ndarray`
            Карта диспаратности
        """
        for idx, frame in enumerate(frames):
            frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)

        disparity_left = self.matcher.compute(*frames).astype(np.int16)
        disparity_right = right_matcher.compute(*frames[::-1]).astype(np.int16)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.matcher)
        wls_filter.setLambda(8000)
        wls_filter.setSigmaColor(1.5)

        return wls_filter.filter(
            disparity_left,
            frames[0],
            disparity_map_right=disparity_right
        )

from typing import Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .models import CalibrationData, RectificationData, TransformationMap

if TYPE_CHECKING:
    from ..datasets.frames import FramesDataset


class StereoCalibrator:
    """
    Класс, вычисляющий внутренние и внешние параметры камер

    Параметры
    ---------
    dataset: :class:`Dataset`
        Папка с изображениями для калибровки
    pattern_size: Tuple[:class:`int`]
        Размер шахматной доски
        Например, 9x6 углов (9, 6)
    square_size: :class:`int`
        Длина квадрата доски в см.
    """
    def __init__(
        self,
        dataset: "FramesDataset",
        pattern_size: Tuple[int],
        square_size: int
    ) -> None:
        self.dataset = dataset
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.img_size = (dataset.images[0].shape[1], dataset.images[0].shape[0])
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        self.pattern_points = [pattern_points] * len(dataset)

    def find_corners(
        self,
        image: cv2.typing.MatLike
    ):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, corners = cv2.findChessboardCorners(gray, self.pattern_size)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

        return corners

    def calibrate(self) -> Tuple[CalibrationData, RectificationData, TransformationMap]:
        left_pts = []
        right_pts = []
        for left_image, right_image in self.dataset:
            left_pts.append(self.find_corners(left_image))
            right_pts.append(self.find_corners(right_image))

        _, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
            self.pattern_points, left_pts, self.img_size, None, None
        )
        _, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
            self.pattern_points, right_pts, self.img_size, None, None
        )

        data = cv2.stereoCalibrate(
            self.pattern_points,
            left_pts,
            right_pts,
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            self.img_size,
            0
        )
        calibration_data = CalibrationData(*data)

        data = cv2.stereoRectify(
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            self.img_size,
            calibration_data.rotation_matrix,
            calibration_data.translation_vector
        )
        rectification_data = RectificationData(*data)

        lmap1, lmap2 = cv2.initUndistortRectifyMap(
            mtx_l,
            dist_l,
            rectification_data.left_rectification_matrix,
            rectification_data.left_projection_matrix,
            self.img_size,
            cv2.CV_16SC2
        )

        rmap1, rmap2 = cv2.initUndistortRectifyMap(
            mtx_r,
            dist_r,
            rectification_data.right_rectification_matrix,
            rectification_data.right_projection_matrix,
            self.img_size,
            cv2.CV_16SC2
        )

        transformation_map = TransformationMap(lmap1, lmap2, rmap1, rmap2)

        return calibration_data, rectification_data, transformation_map

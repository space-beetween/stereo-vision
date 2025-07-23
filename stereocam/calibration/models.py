from dataclasses import dataclass

import numpy as np

from ..mixins import NpzMixin


__all__ = (
    "CalibrationData",
    "RectificationData",
    "TransformationMap",
    "DisparityMap",
)


@dataclass
class CalibrationData(NpzMixin):
    reprojection_error: float
    camera_matrix_left: np.ndarray
    dist_coeffs_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_right: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    essential_matrix: np.ndarray
    fundamental_matrix: np.ndarray


@dataclass
class RectificationData(NpzMixin):
    left_rectification_matrix: np.ndarray
    right_rectification_matrix: np.ndarray
    left_projection_matrix: np.ndarray
    right_projection_matrix: np.ndarray
    disparity_to_depth_matrix: np.ndarray
    left_valid_roi: tuple
    right_valid_roi: tuple


@dataclass
class TransformationMap(NpzMixin):
    left_undistortion_map: np.ndarray
    left_rectification_map: np.ndarray
    right_undistortion_map: np.ndarray
    right_rectification_map: np.ndarray


@dataclass
class DisparityMap(NpzMixin):
    disparity: np.ndarray

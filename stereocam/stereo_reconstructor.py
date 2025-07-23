import cv2
import numpy as np
import open3d
from pathlib import Path

from .calibration import RectificationData
from .types import MapImagesPair


class StereoReconstructor:
    """
    Класс для создания облака точек из карты диспаратности

    Параметры
    ---------
    rectification_data: :class:`RectificationData`
        Данные, полученные после стерео ректификации.
    """
    def __init__(
        self,
        rectification_data: RectificationData,
    ) -> None:
        self.rectification_data = rectification_data

    def save_point_cloud(self, map_images_pair: MapImagesPair, fp: Path) -> None:
        """
        Сохраняет облако точек в .ply файл

        Параметры
        ---------
        map_images_pair: :class:`MapImagesPair`
            Кортеж, где первый элемент - список кадров,
            а второй - карта диспаратности
        """
        frames, disparity_map = map_images_pair
        left_image = frames[0][:disparity_map.shape[0], :disparity_map.shape[1]]
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

        points = cv2.reprojectImageTo3D(
            disparity_map,
            self.rectification_data.disparity_to_depth_matrix
        )

        mask = disparity_map > disparity_map.min()
        points = points[mask]
        colors = left_image[mask]

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.colors = open3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
        open3d.io.write_point_cloud(str(fp), pcd)

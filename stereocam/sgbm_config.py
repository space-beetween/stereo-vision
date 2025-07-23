from typing import Union
from pathlib import Path

import cv2
import yaml


class SGBMConfig():
    def __init__(self, **kwds) -> None:
        self.__dict__.update(kwds)

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "SGBMConfig":
        if isinstance(path, str):
            path = Path(path)
        data = yaml.load(path.read_text("utf-8"), yaml.FullLoader)
        if data is None:
            raise Exception("Конфиг не найден")

        return cls(**data)

    def get_matcher(self, mode: int) -> cv2.StereoMatcher:
        return cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * 1 * self.block_size ** 2,
            P2=32 * 1 * self.block_size ** 2,
            disp12MaxDiff=self.disp_12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            preFilterCap=self.pre_filter_cap,
            mode=mode  # Режим задается программой
        )

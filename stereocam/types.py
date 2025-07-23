from typing import TypeAlias, List, Tuple

import cv2
import numpy as np


MapImagesPair: TypeAlias = Tuple[List[cv2.typing.MatLike], np.ndarray]
ImagePair: TypeAlias = List[cv2.typing.MatLike]

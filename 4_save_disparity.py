import argparse
import threading
import msvcrt
from pathlib import Path

import cv2

from stereocam import StereoPair, DisparityEstimator
from stereocam.calibration import TransformationMap

DESCRIPTION = (
    "Этот скрипт предназначен для показа и сохранения карты диспаратности.\n"
    "Чтобы сохранить карту, в терминале нажмите S.\n"
    "Помимо карты в папку сохраняются исходные кадры с камер.\n"
    "Это необходимо для дальнейшего наложения цветом на облако точек.\n"
    "Поддерживает сохранение нескольких карт диспаратности."
)
MAPS_PATH = 'transformation_map.npz'

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    'cams',
    nargs=2,
    type=int,
    help="Индексы камер"
)
parser.add_argument(
    "output",
    type=str,
    help="Путь к папке, в которую будет сохранена карта диспаратности"
)
args = parser.parse_args()
stereo_pair = StereoPair(args.cams)
output_folder = Path(args.output)
frames_folder = output_folder / 'frames'
frames_folder.mkdir(exist_ok=True)


def save_on_keypress(wait_key: bytes = b's'):
    estimator = DisparityEstimator(TransformationMap.load(MAPS_PATH))
    estimator.set_mode(cv2.STEREO_SGBM_MODE_HH)

    save_count = 0
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == wait_key:
                print("Сохранение карты началось")
                frames = stereo_pair.get_frames()
                stereo_pair.save_frames(
                    str(frames_folder),
                    frames
                )
                stereo_pair.save_disparity(
                    f"{output_folder}/{save_count}.npz",
                    estimator,
                    frames
                )
                print("Карта сохранена")
                save_count += 1


save_thread = threading.Thread(target=save_on_keypress, daemon=True)
save_thread.start()

try:
    stereo_pair.show_disparity_map(DisparityEstimator(TransformationMap.load(MAPS_PATH)))
except KeyboardInterrupt:
    pass

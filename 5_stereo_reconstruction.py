import argparse
from pathlib import Path

from stereocam.datasets import DisparityDataset
from stereocam.calibration import RectificationData
from stereocam import StereoReconstructor


DESCRIPTION = (
    "Этот скрипт предназначен для создания облака точек из карты диспаратности."
)
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    "input_folder",
    type=str,
    help="Путь к папке, в которой сохранены карты диспаратности"
)
args = parser.parse_args()

models_folder = Path("point_clouds")
models_folder.mkdir(exist_ok=True)

dataset = DisparityDataset(args.input_folder)
recification_data = RectificationData.load('rectify.npz')
reconstructor = StereoReconstructor(recification_data)

for idx, map_images_pair in enumerate(dataset):
    reconstructor.save_point_cloud(map_images_pair, models_folder / f'{idx}.ply')

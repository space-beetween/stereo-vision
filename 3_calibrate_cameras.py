import argparse

from stereocam import FramesDataset, StereoCalibrator


DESCRIPTION = (
    "Этот скрипт предназначен для вычисления внешних и внутренних параметров камер\n"
    "На выходе в корневую директорию будет записано три файла\n"
    "calib.npz - результат самой калибровки\n"
    "rectify.npz - результат стерео-ректификации, используются для построения карт ремаппинга\n"
    "transformation_map.npz - карты трансформации (ремапинга)"
)


parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    'folder_path',
    type=str,
    help="Путь к папке с изображениями"
)
parser.add_argument(
    "rows",
    type=int,
    help="Количество внутренних углов в строках шахматной доски.",
)
parser.add_argument(
    "columns",
    type=int,
    help="Количество внутренних углов в столбцах шахматной доски.",
)
parser.add_argument(
    "square_size",
    type=float,
    help="Размер квадрата в см.",
)
args = parser.parse_args()

dataset = FramesDataset(args.folder_path)
calibrator = StereoCalibrator(
    dataset,
    (args.columns, args.rows),
    args.square_size
)
calibration_data, rectification_data, transformation_map = calibrator.calibrate()
calibration_data.save("calib.npz")
rectification_data.save("rectify.npz")
transformation_map.save("transformation_map.npz")

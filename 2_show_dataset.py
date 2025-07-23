import argparse

from stereocam import FramesDataset

DESCRIPTION = (
    "Этот скрипт предназначен для показа изображений датасета.\n",
    "Нужен для окончательной проверки качества изображений\n",
    "Чтобы переключить изображения, нажмите на любую кнопку"
)

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    'folder_path',
    type=str,
    help="Путь к папке с изображениями"
)
args = parser.parse_args()

dataset = FramesDataset(args.folder_path)
dataset.show()

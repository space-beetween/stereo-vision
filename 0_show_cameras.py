import argparse

from stereocam import StereoPair

DESCRIPTION = (
    "Этот скрипт предназначен для показа видеопотока с камер"
)

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    'cams',
    nargs=2,
    type=int,
    help="Индексы камер"
)
args = parser.parse_args()
stereo_pair = StereoPair(args.cams)

try:
    stereo_pair.show_videos()
except KeyboardInterrupt:
    pass

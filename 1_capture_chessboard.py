import time
import argparse
import threading

from stereocam import StereoPair


DESCRIPTION = (
    "Этот скрипт предназначен для захвата шахматной доски.\n"
    "Захват шахматной доски необходим для калибровки по шаблону.\n"
    "Рекомендуется сделать как минимум 30 снимков."
)

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    'cams',
    nargs=2,
    type=int,
    help="Индексы камер"
)
parser.add_argument(
    "amount",
    type=int,
    help="Количество снимков"
)
parser.add_argument(
    "output",
    type=str,
    help="Папка, в которую сохраняются кадры"
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
args = parser.parse_args()

stereo_pair = StereoPair(args.cams)


def save_frames():
    print("Поместите доску так, чтобы ее было видно на обеих камерах")
    for n in range(args.amount):
        time.sleep(5.0)
        frames, _ = stereo_pair.get_chessboard_frames(args.columns, args.rows)
        stereo_pair.save_frames(args.output, frames)
        stereo_pair._saved_frames_count += 1
        print(f"кадры сохранены {n}")
    print("все снимки сделаны")


save_thread = threading.Thread(target=save_frames, daemon=True)
save_thread.start()

try:
    stereo_pair.show_videos()
except KeyboardInterrupt:
    pass

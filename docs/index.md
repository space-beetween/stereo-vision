# Stereo vision

Набор python-скриптов для стереозрения

## Установка
Поддерживается Python 3.10 и новее

Рекомендуется создать виртуальное окружение
```bash
python -m venv .vevv
```

Установите необходимые зависимости
```bash
pip install -r requirements.txt
```

## Структура проекта
    stereocam/                      # Пакет инструментов для работы со стереозрением
    0_show_cameras.py               # Скрипт для отображения двух камер
    1_capture_chessboard.py         # Скрипт для записи изображений для калибровки
    2_show_dataset.py               # Скрипт для отображения записанных изображений
    3_calibrate_cameras.py          # Скрипт для калибровки камер
    4_save_disparity.py             # Скрипт для построения карты диспаратности
    5_stereo_reconstruction.py      # Скрипт для 3д реконструкции
    sgbm_config.yml                 # Файл конфигурации SGBM алгоритма

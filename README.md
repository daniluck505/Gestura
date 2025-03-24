# Gestura

**Gestura** — это приложение на Python, использующее модель YOLO для распознавания жестов руки с камеры и управления курсором мыши. Реализовано перемещение, клики и drag & drop на основе предсказанных классов объектов.

Приложение использует:

- YOLO для детекции жестов.
- OpenCV для работы с изображениями.
- pynput для управления курсором.
- screeninfo для получения информации о разрешении экрана.
- Поддерживает настройку параметров через YAML-файл.

```
/app
├── gestura.py
├── requirements.txt
├── gestura.ipynb
├── config.yml
├── README.md
└── YOLOv10n_gestures.pt
```

## Настройки через YAML

```yaml
model_weights: "YOLOv10n_gestures.pt"
camera_index: 2
buffer_size: 5
scale: 0.2
speed: 0.2
conf: 0.7
iou: 0.5
device: "cpu"
frame_size:
  width: 720
  height: 480
```

## Требования

- Python 3.10+
- Docker (если запускать в контейнере)

## Запуск

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Запустите скрипт:

```bash
python gestura.py
```

## Поддерживаемые жесты

| Класс | Действие                     |
| ---------- | ------------------------------------ |
| 18         | Перемещение мыши      |
| 14         | Левый клик                  |
| 28/29      | Начать drag & drop             |
| 2          | Выход из приложения |

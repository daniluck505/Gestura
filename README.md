# Gestura
**Gestura** — это приложение на Python, использующее модель YOLO для распознавания жестов руки с камеры и управления курсором мыши. Реализовано перемещение, клики и drag & drop на основе предсказанных классов объектов.

![example](https://github.com/user-attachments/assets/90e0f970-cade-4ec4-ab4f-f78c86ff2112)

Приложение использует:
- Python 3.12
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
camera_index: 2           # обычно 0
buffer_size: 5            # повышает сглаживание
scale: 0.5                # парамтр дальности
speed: 0.2                # скорость курсора
conf: 0.7                 # минимальный порог уверенности модели
iou: 0.5                  # iou для алгоритма nms
device: "cpu"             # cpu / cuda / mps (apple silicon)
frame_size:               # разрешение считывания с камеры
  width: 720
  height: 480
model_weights: "YOLOv10n_gestures.pt"
```

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
| 18/21        | Перемещение мыши      |
| 14         | Левый клик                  |
| 22/26         | Правый клик                  |
| 28/29      | Начать drag & drop             |
| 2          | Выход из приложения |
| 23 -> 18   | Вкл/Выкл прокрутки                  |
| 13         | Прокрутка вниз                  |
| 16         | Прокрутка вверх                  |
| 0         | Прокрутка вправо/влево                  |






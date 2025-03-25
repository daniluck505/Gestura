import cv2
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
from collections import deque
import os

class Gestura:
    def __init__(self, config):
        self.model = YOLO(config['model_weights'])
        if config['half']:
            self.model.fuse()
            self.model = self.model.half()
        
        self.camera_index = config['camera_index']
        self.confidence = config['conf']
        self.iou_threshold = config['iou']
        self.device = config['device']
        self.frame_size = (config['frame_size']['width'], config['frame_size']['height'])
        self.mouse_controller = Controller()
        self.scale_factor = config['scale']
        self.movement_speed = config['speed']

        self.buffer = {'x': deque(maxlen=config['buffer_size']), 'y': deque(maxlen=config['buffer_size']),
                       'bbox': deque(maxlen=config['buffer_size']), 'classes': deque(maxlen=config['buffer_size'])}
        monitor = next((m for m in get_monitors() if m.is_primary), get_monitors()[0])
        self.screen_width, self.screen_height = monitor.width, monitor.height
        self.control_area = {'x1': 0, 'y1': 0, 'width': 1, 'height': 1}

        self.clicked = False
        self.dragging = False
        self.exit = False
        self.scroll = False

    def predict_hands(self, image):
        return self.model.predict(image, device=self.device, conf=self.confidence, iou=self.iou_threshold, verbose=False)[0]

    def process_detections(self, detections, image_shape):
        ids_detected = set()

        for detection in detections:
            for bbox, class_id in zip(detection.boxes.xyxy.cpu().int().tolist(), detection.boxes.cls.cpu().int().tolist()):
                x_min, y_min, x_max, y_max = bbox
                center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                self.buffer['bbox'].append(bbox)
                self.buffer['classes'].append(class_id)
                self.buffer['x'].append(center_x)
                self.buffer['y'].append(center_y)
                self._update_control_area(image_shape, bbox)

                ids_detected.add(class_id)

                if class_id in {18, 21}:
                    self._move_cursor()
                    self.clicked = False

                if class_id == 14 and not self.clicked:
                    self._left_click()
                    self.clicked = True

                if class_id in {28, 29}:
                    if not self.dragging:
                        self._start_drag()
                    self._move_cursor()
                
                if class_id in {22, 26} and not self.clicked:
                    self._right_click()
                    self.clicked = True
                    
                if class_id == 23 and not self.clicked:
                    self.scroll = not self.scroll
                    self.clicked = True
                
                if class_id == 13 and self.scroll:
                    self._scroll_down() 
                
                if class_id == 16 and self.scroll:
                    self._scroll_up()
                
                if class_id == 0 and self.scroll:
                    self._swipe_left_or_right(center_x)


        if not ids_detected.intersection({28, 29}) and self.dragging:
            self._stop_drag()

        if 2 in ids_detected:
            self.exit = True

    def _scroll_down(self):
            self.mouse_controller.scroll(0, -2)
        
    def _scroll_up(self):
        self.mouse_controller.scroll(0, 2)

    def _swipe_left_or_right(self, cursor_x):
        if cursor_x < self.frame_size[0] // 2:
            self.mouse_controller.scroll(2, 0)
        else:
            self.mouse_controller.scroll(-2, 0)

    def _update_control_area(self, image_shape, bbox):
        x_min, y_min, x_max, y_max = bbox
        offset_x = int((x_max - x_min) * (1 + self.scale_factor) / 2)
        offset_y = int((y_max - y_min) * (1 + self.scale_factor) / 2)
        self.control_area['x1'] = offset_x
        self.control_area['y1'] = offset_y
        self.control_area['width'] = max(image_shape[1] - 2 * offset_x, 1)
        self.control_area['height'] = max(image_shape[0] - 2 * offset_y, 1)

    def _move_cursor(self):
        avg_x = sum(self.buffer['x']) / len(self.buffer['x'])
        avg_y = sum(self.buffer['y']) / len(self.buffer['y'])
        norm_x = (avg_x - self.control_area['x1']) / self.control_area['width']
        norm_y = (avg_y - self.control_area['y1']) / self.control_area['height']
        target_x = norm_x * self.screen_width
        target_y = norm_y * self.screen_height
        current_x, current_y = self.mouse_controller.position
        new_x = max(0, min(current_x + (target_x - current_x) * self.movement_speed, self.screen_width))
        new_y = max(0, min(current_y + (target_y - current_y) * self.movement_speed, self.screen_height))
        self.mouse_controller.position = (new_x, new_y)

    def _left_click(self):
        self.mouse_controller.click(Button.left)
    
    def _right_click(self):
        self.mouse_controller.click(Button.right)

    def _start_drag(self):
        self.mouse_controller.press(Button.left)
        self.dragging = True

    def _stop_drag(self):
        self.mouse_controller.release(Button.left)
        self.dragging = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError('Не удалось открыть камеру')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])

        try:
            while True and not self.exit:
                success, frame = cap.read()
                if not success:
                    raise RuntimeError('Не удалось получить кадр с камеры')
                frame = cv2.flip(frame, 1)
                detections = self.predict_hands(frame)
                self.process_detections(detections, frame.shape)
        finally:
            cap.release()

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

detector = Gestura(config)
detector.run()
os.system('clear')
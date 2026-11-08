import numpy as np
import cv2
from ultralytics import YOLO
from domain.models import Detection
from domain.ports import IDetector
from config import get_settings

# Tamanho de inferência — maior = mais preciso para objetos pequenos
_IMGSZ = 1280


class YoloDetector(IDetector):
    def __init__(self):
        cfg = get_settings()
        self._model = YOLO(cfg.yolo_model)
        self._confidence = cfg.confidence_threshold
        self._vehicle_classes = cfg.vehicle_classes
        self._device = cfg.device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(
            frame,
            conf=self._confidence,
            device=self._device,
            imgsz=_IMGSZ,
            iou=0.45,       # NMS IoU — reduz duplicatas
            agnostic_nms=True,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in self._vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ignora crops muito pequenos (ruído / veículos muito distantes)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            detections.append(Detection(
                class_id=class_id,
                category=self._vehicle_classes[class_id],
                confidence=float(box.conf[0]),
                bbox=(x1, y1, x2, y2),
                crop=frame[y1:y2, x1:x2],
            ))

        return detections

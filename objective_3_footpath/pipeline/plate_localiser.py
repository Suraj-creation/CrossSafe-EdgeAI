"""
Stage 4 — Licence Plate Localisation using YOLOv8n-LP (TFLite INT8).

Separate fine-tuned YOLOv8n trained to detect a single class: licence_plate.
Runs on the cropped vehicle region from Stage 1 to find the tight
plate bounding box before OCR.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


class PlatLocaliser:
    def __init__(
        self,
        model_path: str = "models/lp_localise_int8.tflite",
        conf_threshold: float = 0.30,
    ):
        self.conf_threshold = conf_threshold

        resolved = Path(model_path)
        if not resolved.exists():
            pt_fallback = Path("models/yolov8n.pt")
            if pt_fallback.exists():
                model_path = str(pt_fallback)
            else:
                model_path = "yolov8n.pt"

        self.model = YOLO(model_path, task="detect")

    def localise(
        self, vehicle_crop: np.ndarray, padding: int = 5
    ) -> Optional[dict]:
        """
        Detect the licence plate within a vehicle crop image.

        Returns the best plate detection dict with keys:
          bbox, confidence, plate_crop
        or None if no plate found.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        results = self.model(vehicle_crop, conf=self.conf_threshold, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = [int(v) for v in boxes[best_idx]]

        h, w = vehicle_crop.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        plate_crop = vehicle_crop[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return None

        return {
            "bbox": [x1, y1, x2, y2],
            "confidence": float(confs[best_idx]),
            "plate_crop": plate_crop,
        }

    def localise_all(self, vehicle_crop: np.ndarray) -> list[dict]:
        """Return all detected plates, sorted by confidence descending."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        results = self.model(vehicle_crop, conf=self.conf_threshold, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        h, w = vehicle_crop.shape[:2]

        plates = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
            x2, y2 = min(w, x2 + 5), min(h, y2 + 5)
            crop = vehicle_crop[y1:y2, x1:x2]
            if crop.size > 0:
                plates.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(confs[i]),
                    "plate_crop": crop,
                })

        plates.sort(key=lambda d: d["confidence"], reverse=True)
        return plates

"""
Stage 1 — Two-Wheeler Detection using YOLOv8n (TFLite INT8).

Detects motorcycles, bicycles, e-scooters, and scooters in camera frames.
Optimised for footpath-mounted camera perspective (3.5–5m height, 25–35° down).
"""

import numpy as np
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


CLASS_NAMES = {0: "motorcycle", 1: "bicycle", 2: "e-scooter", 3: "scooter"}


class TwoWheelerDetector:
    def __init__(
        self,
        model_path: str = "models/twowheeler_int8.tflite",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.50,
        min_bbox_area: int = 1500,
        tracker_config: str = "config/bytetrack.yaml",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_bbox_area = min_bbox_area
        self.tracker_config = tracker_config

        resolved = Path(model_path)
        if not resolved.exists():
            pt_fallback = Path("models/yolov8n.pt")
            if pt_fallback.exists():
                model_path = str(pt_fallback)
            else:
                model_path = "yolov8n.pt"

        self.model = YOLO(model_path, task="detect")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection without tracking. Returns list of detections."""
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        return self._parse_results(results)

    def detect_and_track(self, frame: np.ndarray) -> list[dict]:
        """Run detection with ByteTrack tracking. Returns detections with stable IDs."""
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        return self._parse_results(results, with_tracking=True)

    def _parse_results(
        self, results, with_tracking: bool = False
    ) -> list[dict]:
        if results is None or len(results) == 0:
            return []

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        track_ids = None
        if with_tracking and r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy().astype(int)

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_bbox_area:
                continue

            det = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confs[i]),
                "class_id": int(classes[i]),
                "class_name": CLASS_NAMES.get(int(classes[i]), "unknown"),
                "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                "bottom_center": (int((x1 + x2) / 2), int(y2)),
            }
            if track_ids is not None:
                det["track_id"] = int(track_ids[i])

            detections.append(det)

        return detections

    def warmup(self, imgsz: int = 320):
        """Run a dummy inference to warm up the model."""
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

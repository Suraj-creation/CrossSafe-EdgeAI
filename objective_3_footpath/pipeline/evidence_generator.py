"""
Stage 7 — Evidence Packaging & e-Challan Generation.

For each confirmed violation, creates a complete evidence bundle:
  - Annotated full frame with violation highlighted
  - Raw and enhanced plate crops
  - Structured JSON metadata record
  - Optional MQTT push to police dashboard (non-blocking)
"""

import cv2
import json
import uuid
import threading
import datetime
import os
import numpy as np
from pathlib import Path
from typing import Optional


class EvidenceGenerator:
    def __init__(
        self,
        violations_dir: str = "violations",
        camera_config: Optional[dict] = None,
        dashboard_config: Optional[dict] = None,
    ):
        self.violations_dir = Path(violations_dir)
        self.violations_dir.mkdir(parents=True, exist_ok=True)
        self.cam_cfg = camera_config or {}
        self.dash_cfg = dashboard_config or {}
        self._mqtt_client = None

    def generate(
        self,
        frame: np.ndarray,
        vehicle_crop: Optional[np.ndarray],
        plate_crop_raw: Optional[np.ndarray],
        plate_crop_enhanced: Optional[np.ndarray],
        plate_text: str,
        plate_confidence: float,
        plate_valid: bool,
        vehicle_class: str,
        speed_kmph: float,
        track_id: int,
        bbox: list[float],
        pipeline_latency_ms: Optional[float] = None,
    ) -> dict:
        """Create full evidence package and return the violation record."""

        now = datetime.datetime.now()
        violation_id = str(uuid.uuid4())
        ts_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{ts_str}_{plate_text or 'UNKNOWN'}"
        folder = self.violations_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        annotated = self._annotate_frame(frame, bbox, plate_text, speed_kmph)
        cv2.imwrite(
            str(folder / "evidence_frame.jpg"),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if vehicle_crop is not None and vehicle_crop.size > 0:
            cv2.imwrite(str(folder / "vehicle_crop.jpg"), vehicle_crop)

        if plate_crop_raw is not None and plate_crop_raw.size > 0:
            cv2.imwrite(str(folder / "plate_crop_raw.jpg"), plate_crop_raw)

        if plate_crop_enhanced is not None and plate_crop_enhanced.size > 0:
            cv2.imwrite(str(folder / "plate_crop_enhanced.jpg"), plate_crop_enhanced)

        thumb = cv2.resize(annotated, (320, 240))
        cv2.imwrite(str(folder / "thumbnail.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])

        record = {
            "violation_id": violation_id,
            "timestamp": now.isoformat(),
            "timestamp_epoch": int(now.timestamp()),
            "location": {
                "camera_id": self.cam_cfg.get("camera_id", "FP_CAM_001"),
                "device_id": self.cam_cfg.get("device_id", "EDGE-001"),
                "location_name": self.cam_cfg.get("location_name", "Unknown"),
                "gps_lat": self.cam_cfg.get("gps_lat", 0.0),
                "gps_lng": self.cam_cfg.get("gps_lng", 0.0),
            },
            "vehicle": {
                "plate_number": plate_text,
                "plate_ocr_confidence": round(plate_confidence, 3),
                "plate_format_valid": plate_valid,
                "vehicle_class": vehicle_class,
                "estimated_speed_kmph": speed_kmph,
                "track_id": track_id,
            },
            "violation_type": "FOOTPATH_ENCROACHMENT",
            "section_applied": "Section 177 MV Act / Section 111 BMTC",
            "fine_amount_inr": 500,
            "evidence": {
                "evidence_dir": str(folder),
                "full_frame": str(folder / "evidence_frame.jpg"),
                "vehicle_crop": str(folder / "vehicle_crop.jpg"),
                "plate_crop_raw": str(folder / "plate_crop_raw.jpg"),
                "plate_crop_enhanced": str(folder / "plate_crop_enhanced.jpg"),
                "thumbnail": str(folder / "thumbnail.jpg"),
            },
            "system": {
                "device_id": self.cam_cfg.get("device_id", "EDGE-001"),
                "model_version": "YOLOv8n-v1 + PaddleOCRv3",
                "pipeline_latency_ms": pipeline_latency_ms,
                "pushed_to_dashboard": False,
                "push_timestamp": None,
            },
        }

        with open(folder / "violation_metadata.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        if self.dash_cfg.get("enable_mqtt_push", False):
            threading.Thread(
                target=self._push_mqtt, args=(record,), daemon=True
            ).start()

        return record

    def log_manual_review(
        self,
        plate_text: str,
        plate_confidence: float,
        speed_kmph: float,
        vehicle_class: str,
    ):
        """Log unvalidated plate readings for human review."""
        log_file = self.violations_dir / "manual_review_queue.jsonl"
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "raw_plate": plate_text,
            "confidence": round(plate_confidence, 3),
            "speed_kmph": speed_kmph,
            "vehicle_class": vehicle_class,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _annotate_frame(
        self,
        frame: np.ndarray,
        bbox: list[float],
        plate_text: str,
        speed_kmph: float,
    ) -> np.ndarray:
        annotated = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in bbox]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        label = f"VIOLATION: {plate_text or '???'} | {speed_kmph} km/h"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 16), (x1 + tw + 10, y1), (0, 0, 255), -1)
        cv2.putText(
            annotated, label, (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            annotated, ts, (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

        return annotated

    def _push_mqtt(self, record: dict):
        try:
            import paho.mqtt.publish as publish

            payload = json.dumps({
                "violation_id": record["violation_id"],
                "timestamp": record["timestamp"],
                "plate": record["vehicle"]["plate_number"],
                "speed_kmph": record["vehicle"]["estimated_speed_kmph"],
                "location": record["location"]["location_name"],
                "gps": [record["location"]["gps_lat"], record["location"]["gps_lng"]],
                "fine_inr": record["fine_amount_inr"],
            })

            publish.single(
                self.dash_cfg.get("mqtt_topic", "footpath/violations"),
                payload=payload,
                hostname=self.dash_cfg.get("mqtt_broker_host", "localhost"),
                port=self.dash_cfg.get("mqtt_broker_port", 1883),
                qos=self.dash_cfg.get("mqtt_qos", 1),
            )
            record["system"]["pushed_to_dashboard"] = True
            record["system"]["push_timestamp"] = datetime.datetime.now().isoformat()
        except Exception:
            pass

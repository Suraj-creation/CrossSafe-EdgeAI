"""
Stage 2 — ROI-Based Footpath Boundary Check (No ML).

Determines whether a detected vehicle is inside the footpath region
using geometric point-in-polygon test. Set once at installation,
stored in config JSON. Executes in <1ms.
"""

import cv2
import json
import numpy as np
from pathlib import Path


class ROIChecker:
    def __init__(self, config_path: str = "config/footpath_roi.json"):
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.roi_polygon = np.array(cfg["footpath_roi"], dtype=np.int32)
        self.buffer_px = cfg.get("buffer_zone_expand_px", 15)
        self.camera_id = cfg.get("camera_id", "UNKNOWN")
        self.location = cfg.get("location_name", "UNKNOWN")
        self.gps = (cfg.get("gps_lat", 0.0), cfg.get("gps_lng", 0.0))
        self.frame_size = (
            cfg.get("frame_width", 1920),
            cfg.get("frame_height", 1080),
        )

    def is_on_footpath(
        self,
        bbox: list[float],
        use_bottom_center: bool = True,
    ) -> bool:
        """
        Check if the ground-contact point of a vehicle bbox falls
        inside the footpath ROI polygon.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        if use_bottom_center:
            test_point = (int((x1 + x2) / 2), int(y2))
        else:
            test_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        result = cv2.pointPolygonTest(self.roi_polygon, test_point, False)
        return result >= 0

    def compute_overlap_ratio(self, bbox: list[float]) -> float:
        """
        Compute what fraction of the vehicle bbox overlaps with
        the footpath ROI. More robust for large or partially
        encroaching vehicles.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        fw, fh = self.frame_size

        roi_mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [self.roi_polygon], 255)

        bbox_mask = np.zeros((fh, fw), dtype=np.uint8)
        bbox_mask[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)] = 255

        intersection = np.logical_and(roi_mask, bbox_mask).sum()
        bbox_area = max((x2 - x1) * (y2 - y1), 1)
        return float(intersection / bbox_area)

    def draw_roi(self, frame: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """Draw the ROI polygon on a frame for visualization."""
        overlay = frame.copy()
        cv2.polylines(overlay, [self.roi_polygon], True, color, thickness)
        alpha = 0.15
        fill = frame.copy()
        cv2.fillPoly(fill, [self.roi_polygon], color)
        return cv2.addWeighted(fill, alpha, overlay, 1 - alpha, 0)

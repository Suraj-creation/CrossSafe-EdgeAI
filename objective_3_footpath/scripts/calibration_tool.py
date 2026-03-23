"""
Interactive ROI & Speed Calibration Tool.

Run ONCE at camera installation to:
  1. Draw the footpath ROI polygon on a reference frame
  2. Calibrate pixel-to-metre ratio for speed estimation

Outputs:
  config/footpath_roi.json
  config/speed_calibration.json

Usage:
  python scripts/calibration_tool.py --source rtsp://camera_ip/stream
  python scripts/calibration_tool.py --source 0          # USB camera
  python scripts/calibration_tool.py --source test.jpg   # static image
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class ROICalibrator:
    def __init__(self):
        self.roi_points: list[list[int]] = []
        self.speed_points: list[list[int]] = []
        self.mode = "roi"
        self.frame = None
        self.display = None

    def mouse_handler(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.mode == "roi":
            self.roi_points.append([x, y])
            cv2.circle(self.display, (x, y), 6, (0, 255, 0), -1)
            if len(self.roi_points) > 1:
                cv2.line(
                    self.display,
                    tuple(self.roi_points[-2]),
                    tuple(self.roi_points[-1]),
                    (0, 255, 0), 2,
                )
            cv2.imshow("Calibration", self.display)

        elif self.mode == "speed":
            self.speed_points.append([x, y])
            cv2.circle(self.display, (x, y), 6, (255, 0, 0), -1)
            if len(self.speed_points) == 2:
                cv2.line(
                    self.display,
                    tuple(self.speed_points[0]),
                    tuple(self.speed_points[1]),
                    (255, 0, 0), 2,
                )
            cv2.imshow("Calibration", self.display)

    def capture_frame(self, source) -> np.ndarray:
        if source.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            frame = cv2.imread(source)
            if frame is None:
                print(f"[ERROR] Cannot read image: {source}")
                sys.exit(1)
            return frame

        try:
            src = int(source)
        except ValueError:
            src = source

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {source}")
            sys.exit(1)

        print("[INFO] Press SPACE to capture reference frame, Q to quit...")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                cap.release()
                cv2.destroyWindow("Capture")
                return frame
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    def calibrate_roi(self) -> list[list[int]]:
        self.mode = "roi"
        self.roi_points = []
        self.display = self.frame.copy()

        h, w = self.frame.shape[:2]
        cv2.putText(
            self.display,
            "Click to define footpath polygon. Press ENTER when done, R to reset.",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

        cv2.imshow("Calibration", self.display)
        cv2.setMouseCallback("Calibration", self.mouse_handler)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13:  # ENTER
                if len(self.roi_points) >= 3:
                    self.roi_points.append(self.roi_points[0])
                    cv2.line(
                        self.display,
                        tuple(self.roi_points[-2]),
                        tuple(self.roi_points[-1]),
                        (0, 255, 0), 2,
                    )
                    roi_arr = np.array(self.roi_points[:-1], dtype=np.int32)
                    cv2.fillPoly(self.display, [roi_arr], (0, 255, 0, 40))
                    cv2.imshow("Calibration", self.display)
                    cv2.waitKey(1000)
                    break
                else:
                    print("[WARN] Need at least 3 points for a polygon.")
            elif key == ord("r"):
                self.roi_points = []
                self.display = self.frame.copy()
                cv2.putText(
                    self.display,
                    "Click to define footpath polygon. Press ENTER when done, R to reset.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
                cv2.imshow("Calibration", self.display)

        return self.roi_points

    def calibrate_speed(self) -> float:
        self.mode = "speed"
        self.speed_points = []
        self.display = self.frame.copy()

        cv2.putText(
            self.display,
            "Click TWO endpoints of a 1-metre reference on footpath. Press ENTER when done.",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )
        cv2.imshow("Calibration", self.display)
        cv2.setMouseCallback("Calibration", self.mouse_handler)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13 and len(self.speed_points) >= 2:
                break
            elif key == ord("r"):
                self.speed_points = []
                self.display = self.frame.copy()
                cv2.putText(
                    self.display,
                    "Click TWO endpoints of a 1-metre reference. Press ENTER when done.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                )
                cv2.imshow("Calibration", self.display)

        p1, p2 = self.speed_points[:2]
        pixel_dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        print(f"  Pixel distance for 1 metre: {pixel_dist:.1f} px")
        return float(pixel_dist)

    def run(self, source: str, camera_id: str, location: str,
            gps_lat: float, gps_lng: float, fps: float):
        print("[STEP 1] Capturing reference frame...")
        self.frame = self.capture_frame(source)
        h, w = self.frame.shape[:2]

        ref_path = Path("config/reference_frame.jpg")
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ref_path), self.frame)
        print(f"  Reference frame saved: {ref_path}")

        print("\n[STEP 2] Define footpath ROI polygon...")
        roi_points = self.calibrate_roi()
        print(f"  ROI polygon: {len(roi_points)} points")

        print("\n[STEP 3] Calibrate speed reference (1-metre marker)...")
        ppm = self.calibrate_speed()

        cv2.destroyAllWindows()

        roi_config = {
            "footpath_roi": roi_points,
            "buffer_zone_expand_px": 15,
            "camera_id": camera_id,
            "device_id": "EDGE-001",
            "location_name": location,
            "gps_lat": gps_lat,
            "gps_lng": gps_lng,
            "calibration_date": datetime.now().strftime("%Y-%m-%d"),
            "frame_width": w,
            "frame_height": h,
        }
        roi_path = Path("config/footpath_roi.json")
        with open(roi_path, "w") as f:
            json.dump(roi_config, f, indent=2)
        print(f"\n[SAVED] ROI config → {roi_path}")

        speed_config = {
            "pixels_per_metre": round(ppm, 1),
            "camera_fps": fps,
            "calibration_date": datetime.now().strftime("%Y-%m-%d"),
            "reference_object_length_m": 1.0,
            "reference_object_pixel_length": round(ppm, 1),
        }
        speed_path = Path("config/speed_calibration.json")
        with open(speed_path, "w") as f:
            json.dump(speed_config, f, indent=2)
        print(f"[SAVED] Speed config → {speed_path}")

        print("\n[DONE] Calibration complete!")


def main():
    parser = argparse.ArgumentParser(description="ROI & Speed Calibration Tool")
    parser.add_argument("--source", default="0", help="Camera source (RTSP URL, device index, or image)")
    parser.add_argument("--camera-id", default="FP_CAM_001")
    parser.add_argument("--location", default="MG Road Junction - Footpath North")
    parser.add_argument("--gps-lat", type=float, default=12.9716)
    parser.add_argument("--gps-lng", type=float, default=77.5946)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    calibrator = ROICalibrator()
    calibrator.run(
        args.source, args.camera_id, args.location,
        args.gps_lat, args.gps_lng, args.fps,
    )


if __name__ == "__main__":
    main()

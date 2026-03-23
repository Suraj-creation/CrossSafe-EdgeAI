"""
Objective 3 — Footpath Violation Detection & Auto-Enforcement
=============================================================
Full production inference loop for Raspberry Pi 4 / Jetson Nano.

7-Stage Pipeline:
  1. Two-Wheeler Detection (YOLOv8n TFLite INT8)
  2. ROI Footpath Boundary Check (geometry, no ML)
  3. Multi-Object Tracking + Speed Estimation (ByteTrack)
  4. Licence Plate Localisation (YOLOv8n-LP TFLite INT8)
  5. Plate Image Enhancement (CLAHE + Unsharp Mask, CPU)
  6. OCR — Character Recognition (PaddleOCR PP-OCRv3)
  7. Evidence Packaging + e-Challan Generation + MQTT Push

Usage:
  python main.py
  python main.py --source rtsp://192.168.1.100/stream1
  python main.py --source 0                                # USB camera
  python main.py --source test_video.mp4 --show            # with preview
"""

import argparse
import json
import time
import sys
from pathlib import Path

import cv2
import numpy as np

from pipeline.detector import TwoWheelerDetector
from pipeline.roi_checker import ROIChecker
from pipeline.tracker import VehicleTracker
from pipeline.plate_localiser import PlatLocaliser
from pipeline.plate_enhancer import PlateEnhancer
from pipeline.ocr_engine import IndianPlateOCR
from pipeline.evidence_generator import EvidenceGenerator
from utils.logger import setup_logger
from utils.config_loader import load_config


def load_all_configs(config_dir: str = "config") -> dict:
    """Load all configuration files into a single dict."""
    return {
        "roi": load_config(f"{config_dir}/footpath_roi.json"),
        "speed": load_config(f"{config_dir}/speed_calibration.json"),
        "rules": load_config(f"{config_dir}/violation_rules.json"),
        "dashboard": load_config(f"{config_dir}/dashboard.json"),
    }


def build_pipeline(configs: dict, use_gpu: bool = False):
    """Instantiate all 7 pipeline stages."""
    rules = configs["rules"]

    detector = TwoWheelerDetector(
        model_path="models/twowheeler_int8.tflite",
        conf_threshold=rules["detection_confidence_threshold"],
        iou_threshold=rules["nms_iou_threshold"],
        min_bbox_area=rules["min_bbox_area_px"],
        tracker_config="config/bytetrack.yaml",
    )

    roi_checker = ROIChecker(config_path="config/footpath_roi.json")

    tracker = VehicleTracker(
        pixels_per_metre=configs["speed"]["pixels_per_metre"],
        camera_fps=configs["speed"]["camera_fps"],
        speed_threshold_kmph=rules["speed_threshold_kmph"],
        cooldown_seconds=rules["cooldown_seconds"],
        max_history=rules.get("max_track_history", 15),
    )

    plate_localiser = PlatLocaliser(
        model_path="models/lp_localise_int8.tflite",
        conf_threshold=rules["lp_detection_confidence"],
    )

    plate_enhancer = PlateEnhancer(target_width=400)

    ocr = IndianPlateOCR(use_gpu=use_gpu)

    evidence = EvidenceGenerator(
        violations_dir="violations",
        camera_config=configs["roi"],
        dashboard_config=configs["dashboard"],
    )

    return {
        "detector": detector,
        "roi_checker": roi_checker,
        "tracker": tracker,
        "plate_localiser": plate_localiser,
        "plate_enhancer": plate_enhancer,
        "ocr": ocr,
        "evidence": evidence,
    }


def process_frame(frame, pipeline, rules, logger, show=False):
    """Run full 7-stage pipeline on a single frame. Returns list of violations."""
    t0 = time.time()

    # ── Stage 1 + 3: Detection + Tracking ──────────────────────────────
    detections = pipeline["detector"].detect_and_track(frame)
    if not detections:
        return []

    violations = []

    for det in detections:
        track_id = det.get("track_id")
        if track_id is None:
            continue

        bbox = det["bbox"]
        center = det["center"]
        vehicle_class = det["class_name"]

        # ── Stage 2: ROI check ─────────────────────────────────────────
        if not pipeline["roi_checker"].is_on_footpath(bbox):
            continue

        # ── Stage 3: Speed estimation ──────────────────────────────────
        speed = pipeline["tracker"].update(track_id, center)
        if speed < rules["speed_threshold_kmph"]:
            continue

        # ── Cooldown check ─────────────────────────────────────────────
        now = time.time()
        if pipeline["tracker"].is_in_cooldown(track_id, now):
            continue

        # ── Stage 4: Plate localisation ────────────────────────────────
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 20
        veh_crop = frame[
            max(0, y1 - pad): min(frame.shape[0], y2 + pad),
            max(0, x1 - pad): min(frame.shape[1], x2 + pad),
        ]
        plate_result = pipeline["plate_localiser"].localise(veh_crop)
        if plate_result is None:
            logger.debug(f"Track {track_id}: No plate found")
            continue

        plate_crop_raw = plate_result["plate_crop"]

        # ── Stage 5: Plate enhancement ─────────────────────────────────
        plate_enhanced = pipeline["plate_enhancer"].full_pipeline(plate_crop_raw)

        # ── Stage 6: OCR with voting ──────────────────────────────────
        ocr_result = pipeline["ocr"].read_with_voting(
            plate_enhanced,
            n_runs=rules.get("ocr_voting_runs", 3),
        )

        plate_text = ocr_result["cleaned_text"]
        plate_conf = ocr_result["confidence"]
        plate_valid = ocr_result["is_valid"]

        latency_ms = (time.time() - t0) * 1000

        # ── Stage 7: Evidence packaging ────────────────────────────────
        if plate_text and plate_conf >= rules["ocr_confidence_threshold"]:
            if plate_valid:
                record = pipeline["evidence"].generate(
                    frame=frame,
                    vehicle_crop=veh_crop,
                    plate_crop_raw=plate_crop_raw,
                    plate_crop_enhanced=plate_enhanced,
                    plate_text=plate_text,
                    plate_confidence=plate_conf,
                    plate_valid=plate_valid,
                    vehicle_class=vehicle_class,
                    speed_kmph=speed,
                    track_id=track_id,
                    bbox=bbox,
                    pipeline_latency_ms=latency_ms,
                )
                pipeline["tracker"].record_challan(track_id, now)
                violations.append(record)

                logger.info(
                    f"CHALLAN | {plate_text} | {vehicle_class} | "
                    f"{speed} km/h | conf={plate_conf:.2f} | {latency_ms:.0f}ms"
                )
            else:
                pipeline["evidence"].log_manual_review(
                    plate_text, plate_conf, speed, vehicle_class,
                )
                logger.warning(
                    f"INVALID FORMAT | {plate_text} | conf={plate_conf:.2f} — logged for review"
                )

    return violations


def run(args):
    logger = setup_logger("obj3", "logs")
    logger.info("=" * 60)
    logger.info("Objective 3 — Footpath Violation Detection Starting")
    logger.info("=" * 60)

    configs = load_all_configs(args.config_dir)
    rules = configs["rules"]
    skip_frames = rules.get("skip_frames", 2)

    logger.info("Building pipeline...")
    pipeline = build_pipeline(configs, use_gpu=args.gpu)

    logger.info("Warming up detector...")
    pipeline["detector"].warmup()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {args.source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info(f"Video source opened: {args.source}")
    logger.info(f"Skip frames: {skip_frames} (process 1 in {skip_frames + 1})")

    frame_count = 0
    total_violations = 0
    fps_counter = 0
    fps_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and not source.startswith("rtsp"):
                    logger.info("End of video file.")
                    break
                logger.warning("Frame read failed. Reconnecting...")
                time.sleep(1.0)
                cap.release()
                cap = cv2.VideoCapture(source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            frame_count += 1
            if frame_count % (skip_frames + 1) != 0:
                continue

            violations = process_frame(frame, pipeline, rules, logger, args.show)
            total_violations += len(violations)

            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 5.0:
                fps = fps_counter / elapsed
                logger.info(
                    f"PERF | Frame {frame_count} | {fps:.1f} FPS | "
                    f"Total violations: {total_violations}"
                )
                fps_counter = 0
                fps_start = time.time()

            if args.show:
                display = pipeline["roi_checker"].draw_roi(frame)
                cv2.imshow("Obj3 — Footpath Enforcement", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        logger.info(f"Session ended. Total violations: {total_violations}")


def main():
    parser = argparse.ArgumentParser(
        description="Objective 3 — Footpath Violation Detection & Auto-Enforcement"
    )
    parser.add_argument(
        "--source", default="rtsp://192.168.1.100/stream1",
        help="Video source: RTSP URL, device index (0), or file path",
    )
    parser.add_argument("--config-dir", default="config", help="Config directory")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (Jetson)")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()

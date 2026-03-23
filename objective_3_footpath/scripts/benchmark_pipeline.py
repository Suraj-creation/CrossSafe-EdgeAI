"""
Benchmark end-to-end pipeline latency on the actual deployment device.

Measures FPS, per-frame latency, and memory usage.
Run this ON the edge device (Pi 4 / Jetson) after deploying models.

Usage:
  python scripts/benchmark_pipeline.py --source 0 --frames 200
  python scripts/benchmark_pipeline.py --source test_video.mp4 --frames 500
"""

import argparse
import time
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.detector import TwoWheelerDetector
from pipeline.roi_checker import ROIChecker
from pipeline.plate_localiser import PlatLocaliser
from pipeline.plate_enhancer import PlateEnhancer
from pipeline.ocr_engine import IndianPlateOCR
from utils.config_loader import load_config


def get_memory_mb() -> float:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def run_benchmark(source, num_frames: int, use_gpu: bool = False):
    print("=" * 60)
    print("PIPELINE LATENCY BENCHMARK")
    print("=" * 60)

    mem_before = get_memory_mb()
    print(f"  Memory before loading: {mem_before:.1f} MB")

    print("\n[LOADING] Models...")
    t_load = time.time()

    detector = TwoWheelerDetector()
    plate_loc = PlatLocaliser()
    enhancer = PlateEnhancer()
    roi = ROIChecker()
    ocr = IndianPlateOCR(use_gpu=use_gpu)

    load_time = time.time() - t_load
    mem_after = get_memory_mb()
    print(f"  Model load time: {load_time:.2f}s")
    print(f"  Memory after loading: {mem_after:.1f} MB (+{mem_after - mem_before:.1f} MB)")

    print("\n[WARMUP]...")
    detector.warmup()

    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    latencies = []
    detection_times = []
    plate_times = []
    ocr_times = []
    detections_per_frame = []

    print(f"\n[BENCHMARK] Processing {num_frames} frames...")
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        t_total = time.time()

        t0 = time.time()
        dets = detector.detect(frame)
        detection_times.append((time.time() - t0) * 1000)
        detections_per_frame.append(len(dets))

        if dets:
            det = dets[0]
            bbox = det["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            veh_crop = frame[max(0, y1):y2, max(0, x1):x2]

            if veh_crop.size > 0:
                t0 = time.time()
                plate = plate_loc.localise(veh_crop)
                plate_times.append((time.time() - t0) * 1000)

                if plate:
                    enhanced = enhancer.enhance(plate["plate_crop"])
                    t0 = time.time()
                    ocr.read_plate(enhanced)
                    ocr_times.append((time.time() - t0) * 1000)

        latencies.append((time.time() - t_total) * 1000)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")

    cap.release()

    lat = np.array(latencies)
    det_t = np.array(detection_times)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\n  Frames processed:    {len(latencies)}")
    print(f"  Effective FPS:       {1000 / lat.mean():.1f}")
    print()
    print(f"  {'Metric':<30} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8}")
    print("  " + "-" * 60)
    print(f"  {'Total latency (ms)':<30} {lat.mean():>8.1f} {np.median(lat):>8.1f} "
          f"{np.percentile(lat, 95):>8.1f} {lat.max():>8.1f}")
    print(f"  {'Detection (ms)':<30} {det_t.mean():>8.1f} {np.median(det_t):>8.1f} "
          f"{np.percentile(det_t, 95):>8.1f} {det_t.max():>8.1f}")

    if plate_times:
        pt = np.array(plate_times)
        print(f"  {'Plate localise (ms)':<30} {pt.mean():>8.1f} {np.median(pt):>8.1f} "
              f"{np.percentile(pt, 95):>8.1f} {pt.max():>8.1f}")

    if ocr_times:
        ot = np.array(ocr_times)
        print(f"  {'OCR (ms)':<30} {ot.mean():>8.1f} {np.median(ot):>8.1f} "
              f"{np.percentile(ot, 95):>8.1f} {ot.max():>8.1f}")

    dpf = np.array(detections_per_frame)
    print(f"\n  Avg detections/frame: {dpf.mean():.1f}")
    print(f"  Peak memory usage:    {get_memory_mb():.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline latency")
    parser.add_argument("--source", default="0", help="Video source")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to process")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (Jetson)")
    args = parser.parse_args()

    run_benchmark(args.source, args.frames, args.gpu)


if __name__ == "__main__":
    main()

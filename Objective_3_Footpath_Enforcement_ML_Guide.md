# Objective 3 — Footpath Violation Detection & Auto-Enforcement
## Complete ML Implementation Guide for Edge Deployment

> **System Goal:** Deploy a camera-based edge AI system on footpath-mounted cameras that autonomously detects two-wheelers (motorcycles, bicycles, e-scooters) encroaching on pedestrian footpaths, estimates their speed, extracts the vehicle licence plate via OCR, and generates a geo-tagged e-Challan with photo evidence — all processing done entirely on-device with zero cloud dependency.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Hardware Requirements & Edge Device Selection](#2-hardware-requirements--edge-device-selection)
3. [The Complete ML Pipeline — Stage by Stage](#3-the-complete-ml-pipeline--stage-by-stage)
4. [Stage 1 — Two-Wheeler Detection Model](#4-stage-1--two-wheeler-detection-model)
5. [Stage 2 — ROI-Based Footpath Boundary Logic](#5-stage-2--roi-based-footpath-boundary-logic)
6. [Stage 3 — Multi-Object Tracking & Speed Estimation](#6-stage-3--multi-object-tracking--speed-estimation)
7. [Stage 4 — Licence Plate Localisation Model](#7-stage-4--licence-plate-localisation-model)
8. [Stage 5 — Plate Super-Resolution (Optional but Recommended)](#8-stage-5--plate-super-resolution-optional-but-recommended)
9. [Stage 6 — OCR Engine for Indian Licence Plates](#9-stage-6--ocr-engine-for-indian-licence-plates)
10. [Stage 7 — Evidence Packaging & e-Challan Generation](#10-stage-7--evidence-packaging--e-challan-generation)
11. [Datasets — Complete Curated List](#11-datasets--complete-curated-list)
12. [Model Training — Step-by-Step](#12-model-training--step-by-step)
13. [Edge Conversion & Quantisation](#13-edge-conversion--quantisation)
14. [Full Inference Loop Code](#14-full-inference-loop-code)
15. [Evaluation & Acceptance Criteria](#15-evaluation--acceptance-criteria)
16. [Deployment Checklist](#16-deployment-checklist)
17. [File & Folder Structure](#17-file--folder-structure)
18. [Development Timeline](#18-development-timeline)

---

## 1. System Architecture Overview

### What the System Does — End to End

```
[Footpath Camera — 1080p IP Camera]
            |
            v
[Frame Capture @ 15–25 FPS — OpenCV RTSP / USB]
            |
            v
[STAGE 1]  Two-Wheeler Detection
           Model: YOLOv8n (fine-tuned, TFLite INT8)
           Classes: motorcycle, bicycle, e-scooter, scooter
           Output: Bounding boxes + class labels + confidence scores
            |
            v
[STAGE 2]  ROI Footpath Boundary Check
           Logic: Polygon point-in-region test (deterministic, no ML)
           Output: INSIDE_FOOTPATH flag per detection
            |
            | — if NOT inside footpath → discard, continue loop
            v
[STAGE 3]  Multi-Object Tracking + Speed Estimation
           Model: ByteTrack (lightweight, edge-optimised)
           Logic: Track bbox displacement across frames → speed (km/h)
           Output: Tracked ID + speed value per vehicle
            |
            | — if speed < 5 km/h → classify as parked → skip OCR
            v
[STAGE 4]  Licence Plate Localisation
           Model: YOLOv8n-LP (fine-tuned on Indian LP dataset)
           Input: Cropped vehicle region (from Stage 1 bbox)
           Output: Tight plate bounding box within vehicle crop
            |
            v
[STAGE 5]  Plate Image Enhancement
           Model: ESRGAN-tiny TFLite (upscale 2× or 4×)
           OR:    OpenCV CLAHE + Unsharp Mask (if no GPU)
           Output: Enhanced plate image (128×512 minimum)
            |
            v
[STAGE 6]  OCR — Character Recognition
           Model: PaddleOCR PP-OCRv3 (TFLite / ONNX)
           Post-process: Regex validation for Indian LP format
           Output: Plate string e.g. "KA05AB1234"
            |
            v
[STAGE 7]  Evidence Package Generation
           Logic: Annotate frame + crop plate + write JSON + push alert
           Output: e-Challan JSON + Evidence image + Police dashboard push
```

### Design Principles

- **Edge-First**: Every stage runs on-device. No frame is sent to the cloud.
- **Zero Dependency**: System works fully offline. Internet used only for dashboard push when available.
- **Modular**: Each stage can be swapped independently. E.g., PaddleOCR can replace TesseractOCR without touching the rest.
- **Fail-Safe**: If any stage fails (model crash, timeout), the system logs the raw frame and continues. It never blocks the pipeline.
- **Calibration-Once**: ROI polygon and pixel-to-metre ratio are set once at installation and stored in config JSON.

---

## 2. Hardware Requirements & Edge Device Selection

### Primary Deployment Targets

| Device | RAM | Storage | GPU/NPU | Cost (INR) | Recommended For |
|---|---|---|---|---|---|
| **Raspberry Pi 4 (4GB)** | 4 GB | 64 GB SD + optional SSD | VideoCore VI (no CUDA) | ~5,500 | Primary deployment — cost-effective |
| **NVIDIA Jetson Nano (4GB)** | 4 GB | 64 GB SD | 128-core Maxwell GPU | ~8,000 | High accuracy + speed needed |
| **Raspberry Pi 5 (8GB)** | 8 GB | 64 GB SD | VideoCore VII | ~7,500 | Best Pi option — 2× Pi 4 speed |
| **Orange Pi 5** | 8 GB | 64 GB SD | Mali-G610 + NPU | ~6,500 | Alternative to Jetson, cheaper |

> **Recommendation**: Deploy with **Raspberry Pi 4 (4GB)** for cost-constrained rollout. Use **Jetson Nano** at high-priority intersections or where plate reading in low light is critical (TensorRT acceleration makes a significant difference for OCR quality).

### Camera Requirements

| Parameter | Minimum | Recommended |
|---|---|---|
| Resolution | 1080p (1920×1080) | 2MP–4MP |
| Frame Rate | 15 FPS | 25 FPS |
| Night Vision | IR LEDs (8–10m range) | IR + Starlight CMOS sensor |
| Weatherproofing | IP65 | IP66 |
| Lens | 4mm fixed | 2.8–12mm varifocal |
| Interface | USB / RTSP over Ethernet | RTSP over PoE |
| Shutter | Rolling | **Global Shutter preferred** (avoids motion blur on fast bikes) |

> **Critical Note on Global Shutter**: Two-wheelers moving at 20–30 km/h will produce severe motion blur on rolling-shutter cameras. A global-shutter camera (e.g., Arducam IMX296) is strongly recommended for the plate recognition module to achieve acceptable OCR accuracy.

### Recommended Accessories

- **Google Coral USB Accelerator** (~₹4,000): Plugs into Pi via USB3, provides 4 TOPS for TFLite models. Reduces YOLOv8n inference from ~80ms to ~25ms on Raspberry Pi 4.
- **Hailo-8 M.2 Hat (Pi 5 only)**: 26 TOPS, fastest edge option for Pi platform.
- **IR Illuminator (850nm, 10m range)**: Mandatory for night-time plate reading.
- **PoE Hat for Pi**: Single-cable installation (power + network over Ethernet).

---

## 3. The Complete ML Pipeline — Stage by Stage

### Overview of All Models

| Stage | Task | Model | Framework | Size (Edge) | Latency (Pi 4) | Latency (Jetson) |
|---|---|---|---|---|---|---|
| 1 | Two-wheeler detection | YOLOv8n (fine-tuned) | TFLite INT8 | 3.2 MB | ~55ms @ 320px | ~22ms @ 640px |
| 2 | ROI footpath test | Rule-based polygon | Python/NumPy | 0 MB | <1ms | <1ms |
| 3 | Tracking + speed | ByteTrack | Python | ~0.5 MB | ~5ms | ~3ms |
| 4 | Plate localisation | YOLOv8n-LP | TFLite INT8 | 3.2 MB | ~40ms | ~18ms |
| 5 | Plate enhancement | CLAHE (CPU) / ESRGAN-tiny | OpenCV / TFLite | 2.1 MB | ~8ms | ~5ms |
| 6 | Plate OCR | PaddleOCR PP-OCRv3 | ONNX Runtime | ~8 MB | ~90ms | ~35ms |
| 7 | Evidence generation | Rule-based Python | Python/OpenCV | 0 MB | ~15ms | ~10ms |
| **TOTAL** | | | | **~17 MB** | **~214ms** | **~93ms** |

> **On Raspberry Pi 4**: The pipeline runs end-to-end in ~200–220ms per violation frame. This is acceptable because violation capture only triggers on confirmed violations — it does not need to run on every single frame. The main detection loop runs at 10–15 FPS; the heavy OCR pipeline only activates when a violation is confirmed.

---

## 4. Stage 1 — Two-Wheeler Detection Model

### Model Choice: YOLOv8n (Nano)

**Why YOLOv8n over other options:**

- Ultralytics YOLOv8n is the smallest production-grade YOLO variant at ~6MB (PyTorch) / ~3.2MB (INT8 TFLite).
- COCO pre-trained weights already include `motorcycle` (class 3) and `bicycle` (class 1) — you get a solid baseline before any fine-tuning.
- The Ultralytics Python API makes fine-tuning, export, and inference a single-command workflow.
- Runs at 12–18 FPS on Raspberry Pi 4 at 320px input (sufficient for violation flagging).

**Why NOT larger models:**

- YOLOv8s (~22MB) and above exceed the latency budget on Raspberry Pi 4.
- SSD-MobileNet is faster but less accurate at small/occluded two-wheeler detection.
- YOLO-NAS requires proprietary dependencies that are painful on ARM edge devices.

### Classes to Detect

```
Class 0: motorcycle        (includes sports bikes, cruisers, delivery bikes)
Class 1: bicycle           (includes pedal cycles, cargo bikes)
Class 2: e-scooter         (standalone class — increasingly common in Indian cities)
Class 3: scooter           (gearless scooters — very common in India)
Class 4: auto_rickshaw     (optional — often encroach on footpaths at stops)
```

> **Important:** The base COCO model only has `motorcycle` and `bicycle`. Fine-tuning on Indian street data is mandatory to add `e-scooter` and `scooter` as separate classes, and to improve accuracy on Indian vehicle types (Royal Enfields, Splendors, Activas).

### Confidence & NMS Thresholds

```python
DETECTION_CONFIDENCE_THRESHOLD = 0.45   # Lower than default to catch partial views
NMS_IOU_THRESHOLD               = 0.50
MIN_BBOX_AREA_PX                = 1500   # Ignore tiny detections far from camera
```

---

## 5. Stage 2 — ROI-Based Footpath Boundary Logic

### Concept

The ROI (Region of Interest) is a polygon drawn over the footpath area in the camera frame. It is defined **once at installation time** and stored in `config/footpath_roi.json`. No ML is needed — it is a pure geometric point-in-polygon test.

### ROI Calibration Procedure (Done Once at Installation)

**Step 1:** Capture a clean reference frame from the installed camera.

**Step 2:** Run the calibration tool (provided below) to draw the footpath polygon interactively.

**Step 3:** Save the polygon coordinates to `config/footpath_roi.json`.

**Step 4:** Optionally define a **Buffer Zone** (slightly expanded polygon) for early warning before full violation.

```python
# calibration_tool.py — Run this ONCE during installation
import cv2
import json
import numpy as np

roi_points = []
frame_copy = None

def click_handler(event, x, y, flags, param):
    global roi_points, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append([x, y])
        cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
        if len(roi_points) > 1:
            cv2.line(frame_copy,
                     tuple(roi_points[-2]),
                     tuple(roi_points[-1]),
                     (0, 255, 0), 2)
        cv2.imshow("ROI Calibration", frame_copy)

cap = cv2.VideoCapture("rtsp://camera_ip/stream")  # or 0 for USB
ret, frame = cap.read()
frame_copy = frame.copy()

cv2.imshow("ROI Calibration", frame_copy)
cv2.setMouseCallback("ROI Calibration", click_handler)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

# Close the polygon
roi_points.append(roi_points[0])

# Save to config
config = {
    "footpath_roi": roi_points,
    "buffer_zone_expand_px": 15,      # pixels to expand for buffer zone
    "camera_id": "FP_CAM_001",
    "location_name": "MG Road Junction - Footpath North",
    "gps_lat": 12.9716,
    "gps_lng": 77.5946,
    "calibration_date": "2025-01-01"
}
with open("config/footpath_roi.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"ROI saved with {len(roi_points)} points.")
```

### ROI Violation Check Function

```python
import cv2
import numpy as np

def load_roi(config_path: str) -> np.ndarray:
    import json
    with open(config_path) as f:
        cfg = json.load(f)
    return np.array(cfg["footpath_roi"], dtype=np.int32)

def is_in_footpath(bbox: list, roi: np.ndarray,
                   use_bottom_center: bool = True) -> bool:
    """
    Check if a detected vehicle is inside the footpath ROI.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box in pixel coords
        roi:  numpy array of polygon vertices
        use_bottom_center: if True, use bottom-center of bbox as the test
                           point (more accurate for ground contact point)
    
    Returns:
        True if vehicle is on the footpath
    """
    x1, y1, x2, y2 = bbox
    
    if use_bottom_center:
        test_point = (int((x1 + x2) / 2), int(y2))  # bottom center
    else:
        test_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # center
    
    result = cv2.pointPolygonTest(roi, test_point, measureDist=False)
    return result >= 0  # 1.0 = inside, -1.0 = outside, 0.0 = on edge

def compute_overlap_ratio(bbox: list, roi: np.ndarray) -> float:
    """
    Alternative: compute what fraction of the bbox overlaps with the ROI.
    More robust than single-point test for large vehicles.
    Returns overlap ratio between 0.0 and 1.0.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Create mask for ROI
    frame_h, frame_w = 1080, 1920  # adjust to your resolution
    roi_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi], 255)
    
    # Create mask for bbox
    bbox_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    bbox_mask[y1:y2, x1:x2] = 255
    
    intersection = np.logical_and(roi_mask, bbox_mask).sum()
    bbox_area = (x2 - x1) * (y2 - y1)
    
    return intersection / bbox_area if bbox_area > 0 else 0.0

# Usage
VIOLATION_OVERLAP_THRESHOLD = 0.35  # 35% of vehicle must be on footpath
```

---

## 6. Stage 3 — Multi-Object Tracking & Speed Estimation

### Why Tracking Is Essential

Without tracking, the system would:
- Generate duplicate e-Challans for the same vehicle (one per frame)
- Not be able to estimate speed (speed requires position across multiple frames)
- Not be able to confirm a vehicle was actually **moving** on the footpath (vs parked)

### Tracker Choice: ByteTrack

**Why ByteTrack over DeepSORT:**
- ByteTrack has no re-ID neural network — it is purely IoU-based matching, making it much lighter on the Pi.
- DeepSORT requires a separate appearance descriptor CNN (~50ms extra per tracked object on Pi).
- ByteTrack achieves comparable tracking accuracy to DeepSORT for this use case (fixed camera, non-crowded footpath).
- ByteTrack is natively supported in Ultralytics YOLOv8 — zero extra setup.

```python
from ultralytics import YOLO

# ByteTrack is built into YOLOv8's track() method
model = YOLO('models/twowheeler_int8.tflite', task='detect')

# Run tracking
results = model.track(
    source=frame,
    persist=True,         # maintain track IDs across frames
    tracker="bytetrack.yaml",
    conf=0.45,
    iou=0.50,
)

# Each detection now has: .id (track ID), .boxes.xyxy, .boxes.cls
```

### Speed Estimation from Monocular Camera

Speed is estimated from bounding box displacement across consecutive frames, combined with a pixel-to-metre calibration factor computed at installation.

#### Pixel-to-Metre Calibration (Done Once at Installation)

```python
# calibrate_speed.py
# Place a 1-metre marker (tape measure, known-length object) on the footpath.
# Mark the pixel positions of both ends.

import json

# Measured during installation:
PIXEL_DISTANCE_OF_1_METRE = 47  # pixels (example — measure for your camera)
# This varies by camera angle, focal length, and distance from camera to footpath.

# Save calibration
with open("config/speed_calibration.json", "w") as f:
    json.dump({
        "pixels_per_metre": PIXEL_DISTANCE_OF_1_METRE,
        "camera_fps": 15,
        "calibration_date": "2025-01-01"
    }, f, indent=2)
```

#### Speed Estimation Function

```python
import numpy as np
from collections import defaultdict, deque

# Store position history per track ID
track_history = defaultdict(lambda: deque(maxlen=10))

def update_track_and_estimate_speed(
    track_id: int,
    bbox_center: tuple,
    pixels_per_metre: float,
    fps: float
) -> float:
    """
    Update position history for track_id and return current speed in km/h.
    Returns 0.0 if insufficient history.
    """
    track_history[track_id].append(bbox_center)
    
    if len(track_history[track_id]) < 3:
        return 0.0  # need at least 3 frames for stable speed
    
    # Use last 3 positions for smoothing
    positions = list(track_history[track_id])[-3:]
    
    total_pixel_dist = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        total_pixel_dist += np.sqrt(dx**2 + dy**2)
    
    avg_pixel_dist_per_frame = total_pixel_dist / (len(positions) - 1)
    
    metres_per_frame   = avg_pixel_dist_per_frame / pixels_per_metre
    metres_per_second  = metres_per_frame * fps
    km_per_hour        = metres_per_second * 3.6
    
    return round(km_per_hour, 1)

# Violation trigger rule:
SPEED_THRESHOLD_KMPH = 5.0   # below this = parked (skip e-Challan)
# If speed > 5 km/h AND vehicle is inside ROI → confirmed moving violation
```

---

## 7. Stage 4 — Licence Plate Localisation Model

### Why a Separate Plate Localisation Model

The main two-wheeler detector (Stage 1) finds the whole vehicle. The plate is a tiny sub-region of the vehicle — often <5% of the bounding box area. Running OCR on the full vehicle crop would fail. A dedicated licence plate detector finds the tight plate region before OCR.

### Model Choice: YOLOv8n Fine-Tuned on Indian LP Dataset

This is the same YOLOv8n architecture as Stage 1, but fine-tuned specifically on Indian licence plate images. It is trained to output a single class: `licence_plate`.

```
Input:  Cropped vehicle image (from Stage 1 bbox) — resized to 320×320
Output: [x1, y1, x2, y2] tight bounding box around the plate
```

### Handling Multiple Plates

Indian vehicles can have a front plate and a rear plate. The camera angle determines which is visible. The system selects:
1. The plate with the **highest confidence score** if both are detected.
2. The **largest bounding box** as a tiebreaker.

```python
def get_best_plate(plate_detections: list) -> dict | None:
    """
    From a list of plate detections, return the best one.
    Each detection: {'bbox': [x1,y1,x2,y2], 'conf': float}
    """
    if not plate_detections:
        return None
    # Sort by confidence descending, then by area descending as tiebreaker
    sorted_dets = sorted(
        plate_detections,
        key=lambda d: (d['conf'], (d['bbox'][2]-d['bbox'][0]) * (d['bbox'][3]-d['bbox'][1])),
        reverse=True
    )
    return sorted_dets[0]
```

---

## 8. Stage 5 — Plate Image Enhancement

### The Problem

Indian licence plates captured at distance, in motion, or at night are often:
- Low resolution (the plate occupies only 30–80 pixels in width)
- Motion-blurred (two-wheelers moving at 20+ km/h)
- Glare-affected (headlamps, streetlights reflecting off the plate)
- Poorly contrasted (faded plates, old vehicles, dust-covered plates)

Direct OCR on the raw plate crop produces poor accuracy. Enhancement is critical.

### Option A — CPU-Only Enhancement (Raspberry Pi 4 without Coral)

Use classical OpenCV pipeline. No neural network required. Fast and reliable.

```python
import cv2
import numpy as np

def enhance_plate_cpu(plate_img: np.ndarray,
                      target_width: int = 400) -> np.ndarray:
    """
    Full classical plate enhancement pipeline.
    Works on CPU in <10ms on Raspberry Pi 4.
    """
    # Step 1: Upscale to target width (bicubic interpolation)
    h, w = plate_img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    upscaled = cv2.resize(plate_img, (target_width, new_h),
                          interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Convert to grayscale for processing
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    
    # Step 3: CLAHE — adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    
    # Step 4: Gaussian blur to reduce noise before sharpening
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    
    # Step 5: Unsharp masking — sharpens edges (character strokes)
    sharpened = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)
    
    # Step 6: Bilateral filter — reduce noise, preserve edges
    denoised = cv2.bilateralFilter(sharpened, d=5,
                                   sigmaColor=40, sigmaSpace=40)
    
    # Step 7: Adaptive threshold — improve OCR readability
    # (optional — use only if PaddleOCR has issues with grayscale)
    # thresh = cv2.adaptiveThreshold(denoised, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    
    # Return as BGR (required by PaddleOCR)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def deskew_plate(plate_img: np.ndarray) -> np.ndarray:
    """
    Correct slight rotation/tilt of the plate using Hough line detection.
    Improves OCR accuracy on tilted plates.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=60)
    
    if lines is None:
        return plate_img  # no correction needed
    
    angles = []
    for line in lines[:5]:  # use top 5 strongest lines
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if abs(angle) < 20:  # ignore near-vertical lines
            angles.append(angle)
    
    if not angles:
        return plate_img
    
    median_angle = np.median(angles)
    if abs(median_angle) < 1.0:
        return plate_img  # tilt too small to correct
    
    h, w = plate_img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    deskewed = cv2.warpAffine(plate_img, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return deskewed
```

### Option B — ESRGAN-tiny with TFLite (Jetson Nano Recommended)

Use a lightweight Real-ESRGAN model for neural super-resolution. 4× upscale at significantly higher quality than bicubic.

```python
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

class ESRGANEnhancer:
    def __init__(self, model_path: str = "models/esrgan_tiny.tflite"):
        self.interp = tflite.Interpreter(
            model_path=model_path,
            num_threads=4
        )
        self.interp.allocate_tensors()
        self.input_details  = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()
    
    def enhance(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Super-resolve plate image using ESRGAN-tiny.
        Input:  BGR image, any small size
        Output: BGR image, 4× larger
        """
        # Resize to model's expected input (64×256 for plate aspect ratio)
        inp = cv2.resize(plate_img, (256, 64))
        inp = inp.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=0)  # add batch dim
        
        self.interp.set_tensor(self.input_details[0]['index'], inp)
        self.interp.invoke()
        output = self.interp.get_tensor(self.output_details[0]['index'])
        
        output = np.squeeze(output, axis=0)
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        return output

# Where to get the ESRGAN-tiny TFLite model:
# Source:  github.com/Practical-AI/Real-ESRGAN-tflite
# OR:      Convert from Real-ESRGAN PyTorch → ONNX → TFLite (see Section 13)
# Model size: ~2.1 MB (tiny variant)
# Latency on Jetson Nano: ~25ms per plate crop
# Latency on Raspberry Pi 4: ~180ms (use CPU enhancement for Pi instead)
```

---

## 9. Stage 6 — OCR Engine for Indian Licence Plates

### Why PaddleOCR over Tesseract

| Property | PaddleOCR PP-OCRv3 | Tesseract v5 |
|---|---|---|
| Indian LP accuracy (real-world) | ~92–95% | ~55–70% |
| Speed on Pi (single plate) | ~80–100ms | ~200–400ms |
| Curved / bold text handling | Excellent | Poor |
| Model size | ~8 MB (ONNX) | ~30 MB |
| Edge deployment | ONNX Runtime / TFLite | CPU-only binary |
| Indian character support | Built-in training | Requires custom training |
| Fine-tuning support | Yes (PaddleOCR training pipeline) | Difficult |

**Verdict**: Always use PaddleOCR for this project. Tesseract is not suitable for real-world Indian LP recognition.

### PaddleOCR Integration

```python
# Install: pip install paddlepaddle paddleocr
from paddleocr import PaddleOCR
import re
import numpy as np

class IndianPlateOCR:
    
    VALID_LP_PATTERN = re.compile(
        r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    )
    # Covers formats:
    # KA05AB1234  — most common (state + district + series + number)
    # DL1CAB1234  — Delhi format
    # MH12DE1234  — Maharashtra
    # Also handles BH (Bharat) series: 22BH1234AA
    
    BH_SERIES_PATTERN = re.compile(
        r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$'
    )
    
    def __init__(self, use_gpu: bool = False):
        self.ocr = PaddleOCR(
            use_angle_cls=True,     # correct upside-down plates
            lang='en',
            use_gpu=use_gpu,
            rec_model_dir='models/paddleocr_rec',   # local model cache
            det_model_dir='models/paddleocr_det',
            cls_model_dir='models/paddleocr_cls',
            show_log=False,
        )
    
    def read_plate(self, plate_img: np.ndarray) -> dict:
        """
        Run OCR on an enhanced plate image.
        Returns: {
            'raw_text': str,
            'cleaned_text': str,
            'is_valid': bool,
            'confidence': float
        }
        """
        result = self.ocr.ocr(plate_img, cls=True)
        
        if not result or not result[0]:
            return {'raw_text': '', 'cleaned_text': '', 
                    'is_valid': False, 'confidence': 0.0}
        
        # Concatenate all detected text lines
        all_text = ''
        total_conf = 0.0
        count = 0
        for line in result[0]:
            text, confidence = line[1]
            all_text += text
            total_conf += confidence
            count += 1
        
        avg_conf = total_conf / count if count > 0 else 0.0
        
        # Clean and normalise
        cleaned = self._clean_plate_text(all_text)
        is_valid = self._validate_plate(cleaned)
        
        return {
            'raw_text': all_text,
            'cleaned_text': cleaned,
            'is_valid': is_valid,
            'confidence': round(avg_conf, 3)
        }
    
    def _clean_plate_text(self, raw: str) -> str:
        """Remove spaces, lowercase, common OCR confusions."""
        text = raw.upper().replace(' ', '').replace('-', '').strip()
        
        # Common OCR confusion corrections for LP context:
        # These corrections are position-dependent for Indian LP format
        # Positions 0,1 = state code (letters only)
        # Positions 2,3 = district number (digits only)
        # Positions 4,5 = series letters
        # Positions 6–9 = registration number (digits only)
        
        if len(text) >= 10:
            corrected = list(text)
            # Force digits at positions 2, 3
            for pos in [2, 3]:
                corrected[pos] = self._letter_to_digit(corrected[pos])
            # Force letters at positions 0, 1, 4, 5
            for pos in [0, 1, 4, 5]:
                corrected[pos] = self._digit_to_letter(corrected[pos])
            # Force digits at positions 6–9
            for pos in range(6, min(10, len(corrected))):
                corrected[pos] = self._letter_to_digit(corrected[pos])
            text = ''.join(corrected)
        
        return text
    
    def _letter_to_digit(self, char: str) -> str:
        """Convert commonly confused OCR letters to digits."""
        confusion_map = {'O': '0', 'I': '1', 'l': '1', 
                         'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
        return confusion_map.get(char, char)
    
    def _digit_to_letter(self, char: str) -> str:
        """Convert commonly confused OCR digits to letters."""
        confusion_map = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'}
        return confusion_map.get(char, char)
    
    def _validate_plate(self, text: str) -> bool:
        """Validate against Indian LP formats."""
        return bool(
            self.VALID_LP_PATTERN.match(text) or
            self.BH_SERIES_PATTERN.match(text)
        )


# OCR CONFIDENCE STRATEGY:
# Run OCR 3 times on same plate (with slight augmentation between runs)
# Take the result with highest confidence + valid format match.
# This "majority voting" approach improves accuracy from 85% to 92%+ per plate.

def ocr_with_voting(plate_img: np.ndarray, ocr_engine: IndianPlateOCR,
                    n_runs: int = 3) -> dict:
    """Run OCR multiple times with slight augmentations, take best result."""
    candidates = []
    
    augmentations = [
        lambda x: x,                                  # original
        lambda x: cv2.convertScaleAbs(x, alpha=1.1, beta=10),   # brighter
        lambda x: cv2.convertScaleAbs(x, alpha=0.9, beta=-10),  # darker
    ]
    
    for aug in augmentations[:n_runs]:
        augmented = aug(plate_img)
        result = ocr_engine.read_plate(augmented)
        if result['is_valid']:
            candidates.append(result)
    
    if not candidates:
        # No valid result — return best raw result
        return ocr_engine.read_plate(plate_img)
    
    # Return highest confidence valid result
    return max(candidates, key=lambda x: x['confidence'])
```

### Fine-Tuning PaddleOCR on Indian LP Data (Strongly Recommended)

```bash
# Fine-tuning improves accuracy from ~85% to ~94% on Indian plates

# Step 1: Install PaddlePaddle training environment
pip install paddlepaddle-gpu paddleocr

# Step 2: Download Indian LP recognition dataset (see Section 11)
# Place images in: training_data/indian_lp/
# Create label file: training_data/indian_lp/labels.txt
# Format per line: image_path\tplate_text

# Step 3: Download base PP-OCRv3 rec model weights
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar
tar -xf en_PP-OCRv3_rec_train.tar

# Step 4: Configure fine-tuning
# Edit: configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml
# Set: pretrained_model: ./en_PP-OCRv3_rec_train/best_accuracy
# Set: character_dict_path: ./indian_lp_char_dict.txt  (A-Z, 0-9)
# Set: Train.dataset.data_dir: ./training_data/indian_lp/
# Set: epoch_num: 50
# Set: save_epoch_step: 5

# Step 5: Train
python tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml \
    -o Global.pretrained_model=./en_PP-OCRv3_rec_train/best_accuracy

# Step 6: Export to inference model
python tools/export_model.py \
    -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml \
    -o Global.pretrained_model=./output/rec_ppocr_v3_en/best_accuracy \
       Global.save_inference_dir=./models/paddleocr_rec_indian/

# Step 7: Convert inference model to ONNX for edge deployment
paddle2onnx --model_dir ./models/paddleocr_rec_indian/ \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ./models/paddleocr_rec_indian.onnx \
            --opset_version 11
```

---

## 10. Stage 7 — Evidence Packaging & e-Challan Generation

### What the Evidence Package Contains

For each confirmed violation, the system creates a complete evidence bundle:

```
violations/
  2025-01-15_14-23-07_KA05AB1234/
    evidence_frame.jpg         # Full annotated frame with violation box
    plate_crop_raw.jpg         # Raw plate crop before enhancement
    plate_crop_enhanced.jpg    # Enhanced plate crop used for OCR
    violation_metadata.json    # Full violation record
    thumbnail.jpg              # 320×240 compressed for quick preview
```

### Violation Metadata JSON Schema

```python
import json
import datetime
import uuid

def create_violation_record(
    plate_text: str,
    plate_confidence: float,
    vehicle_class: str,
    speed_kmph: float,
    track_id: int,
    camera_config: dict,
    frame_path: str,
    plate_crop_path: str
) -> dict:
    """Create a complete, structured violation record."""
    
    return {
        "violation_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "timestamp_epoch": int(datetime.datetime.now().timestamp()),
        
        "location": {
            "camera_id": camera_config["camera_id"],
            "location_name": camera_config["location_name"],
            "gps_lat": camera_config["gps_lat"],
            "gps_lng": camera_config["gps_lng"],
        },
        
        "vehicle": {
            "plate_number": plate_text,
            "plate_ocr_confidence": plate_confidence,
            "plate_format_valid": True,
            "vehicle_class": vehicle_class,
            "estimated_speed_kmph": speed_kmph,
            "track_id": track_id,
        },
        
        "violation_type": "FOOTPATH_ENCROACHMENT",
        "section_applied": "Section 177 MV Act / Section 111 BMTC",
        "fine_amount_inr": 500,
        
        "evidence": {
            "full_frame": frame_path,
            "plate_crop_raw": plate_crop_path,
            "plate_crop_enhanced": plate_crop_path.replace("raw", "enhanced"),
            "thumbnail": plate_crop_path.replace("raw.jpg", "thumb.jpg"),
        },
        
        "system": {
            "device_id": camera_config.get("device_id", "EDGE-001"),
            "model_version": "YOLOv8n-v2.1 + PaddleOCRv3",
            "pipeline_latency_ms": None,  # filled by main loop
            "pushed_to_dashboard": False,
            "push_timestamp": None,
        }
    }


def save_violation(record: dict, base_dir: str = "violations/") -> str:
    """Save violation record and return directory path."""
    import os
    
    ts = record["timestamp"].replace(":", "-").replace("T", "_")[:19]
    plate = record["vehicle"]["plate_number"]
    dir_name = f"{ts}_{plate}"
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    
    json_path = os.path.join(dir_path, "violation_metadata.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)
    
    return dir_path
```

### Dashboard Push (MQTT — When Network Available)

```python
import paho.mqtt.client as mqtt
import json
import threading

class DashboardPusher:
    """
    Push violation alerts to police dashboard via MQTT.
    Runs in a background thread — never blocks the main inference loop.
    Falls back to local queue when offline.
    """
    
    BROKER_HOST = "mqtt.policedashboard.local"  # or public broker IP
    BROKER_PORT = 1883
    TOPIC       = "footpath/violations"
    
    def __init__(self):
        self.client = mqtt.Client()
        self.offline_queue = []
        self.connected = False
        
        try:
            self.client.connect(self.BROKER_HOST, self.BROKER_PORT, keepalive=60)
            self.client.loop_start()
            self.connected = True
        except Exception:
            self.connected = False  # offline — queue locally
    
    def push_violation(self, record: dict):
        """Push in background thread. Non-blocking."""
        thread = threading.Thread(
            target=self._push_worker, args=(record,), daemon=True
        )
        thread.start()
    
    def _push_worker(self, record: dict):
        payload = json.dumps({
            "violation_id": record["violation_id"],
            "timestamp": record["timestamp"],
            "plate": record["vehicle"]["plate_number"],
            "speed_kmph": record["vehicle"]["estimated_speed_kmph"],
            "location": record["location"]["location_name"],
            "gps": [record["location"]["gps_lat"], record["location"]["gps_lng"]],
            "fine_inr": record["fine_amount_inr"],
        })
        
        if self.connected:
            try:
                self.client.publish(self.TOPIC, payload, qos=1)
                record["system"]["pushed_to_dashboard"] = True
            except Exception:
                self.offline_queue.append(payload)
        else:
            self.offline_queue.append(payload)  # retry on reconnect
```

---

## 11. Datasets — Complete Curated List

### Dataset Group A — Two-Wheeler & Vehicle Detection

---

#### A1. COCO 2017 (Primary Baseline)

- **URL**: https://cocodataset.org/#download
- **Size**: 118,287 training images
- **Relevant classes**: `motorcycle` (class 3), `bicycle` (class 1)
- **Images with two-wheelers**: ~12,000
- **Format**: COCO JSON (convert to YOLO using `ultralytics` dataset tools)
- **Use for**: Pre-training baseline. YOLOv8n's COCO weights already include these classes — use COCO only to further strengthen the base before fine-tuning on Indian data.
- **Download command**:
  ```bash
  # Using fiftyone (easiest method)
  pip install fiftyone
  python -c "
  import fiftyone.zoo as foz
  dataset = foz.load_zoo_dataset('coco-2017', split='train',
      label_types=['detections'],
      classes=['motorcycle', 'bicycle'],
      max_samples=8000)
  dataset.export(export_dir='datasets/coco_twowheelers/',
      dataset_type=foz.types.YOLOv5Dataset)
  "
  ```

---

#### A2. UA-DETRAC Traffic Dataset

- **URL**: https://detrac-db.rit.albany.edu/
- **Alt URL (Kaggle)**: https://www.kaggle.com/datasets/mdfahimreshm/ua-detrac-datasets
- **Size**: 140,000 frames from 100 traffic sequences
- **Classes**: car, bus, van, motorcycle
- **Annotations**: XML (convert to YOLO using provided scripts)
- **Why useful**: Dense urban traffic scenes, multiple angles, includes Indian-style intersection footage in similar conditions.
- **Note**: Download the motorcycle-heavy sequences only to keep dataset size manageable.

---

#### A3. Indian Driving Dataset (IDD)

- **URL**: https://idd.insaan.iiit.ac.in/
- **Size**: 10,004 images from Indian roads (Hyderabad, Bangalore)
- **Classes**: 26 Indian-specific classes including `two-wheeler`, `auto-rickshaw`, `e-rickshaw`
- **Format**: PASCAL VOC XML (convert to YOLO)
- **Why critical**: This is THE most important dataset for this project. It contains Indian road scenes, Indian vehicles, Indian road markings, and Indian lighting conditions. No international dataset can replace this.
- **Download**: Free registration required at idd.insaan.iiit.ac.in
- **Special note**: IDD contains a separate `two-wheeler` class that already covers motorcycles, scooters, and bicycles in Indian context — far better than COCO's generic motorcycle class.

---

#### A4. DAWN (Adverse Conditions Dataset)

- **URL**: https://paperswithcode.com/dataset/dawn
- **Kaggle**: https://www.kaggle.com/datasets/amlanpraharaj/dawn-dataset
- **Size**: 1,000 images in fog, rain, snow, sand
- **Why useful**: Indian footpath cameras operate in monsoon rain and dusty conditions. DAWN provides adverse condition training data to make the model robust to weather.
- **Use for**: Augmenting the training set with adverse condition examples (add ~200–300 images per weather type).

---

#### A5. BDD100K (Berkeley DeepDrive)

- **URL**: https://bdd-data.berkeley.edu/
- **Size**: 100,000 video frames, fully annotated
- **Classes**: includes `motorcycle`, `bicycle`, plus time-of-day and weather labels
- **Why useful**: Large-scale, diverse, includes night-time and rain conditions. Good for diversity augmentation.
- **Download**: Free registration at bdd-data.berkeley.edu

---

#### A6. Roboflow Universe — Two-Wheeler Detection (Indian)

- **URL**: https://universe.roboflow.com
- **Search queries to use**:
  - `"two wheeler detection india"`
  - `"motorcycle detection footpath"`
  - `"indian traffic two wheeler"`
  - `"scooter detection"`
- **Expected results**: 3–8 datasets, each ~500–3,000 images
- **Format**: Export directly in YOLOv8 format (Roboflow handles conversion)
- **Why essential**: Community-annotated Indian-specific datasets, often with footpath/pavement context already present.

---

#### Dataset A — Final Combined Strategy

```
Total training images for two-wheeler detection:
  COCO 2017 motorcycle/bicycle subset:     ~8,000  images
  IDD (Indian Driving Dataset):            ~5,000  images (two-wheeler class)
  UA-DETRAC motorcycle sequences:          ~4,000  images
  BDD100K motorcycle/bicycle subset:       ~6,000  images
  Roboflow Indian two-wheeler sets:        ~3,000  images
  DAWN adverse conditions (2-wheeler):     ~1,000  images
  Self-collected local street images:      ~500    images (manual annotation)
  ─────────────────────────────────────────────────────
  TOTAL:                                  ~27,500  images

Split: 75% train / 15% val / 10% test
```

---

### Dataset Group B — Licence Plate Localisation

---

#### B1. Open Images V7 — Licence Plate Class

- **URL**: https://storage.googleapis.com/openimages/web/index.html
- **Relevant class**: `Vehicle registration plate`
- **Size**: ~10,000+ annotated images
- **Download**:
  ```bash
  pip install fiftyone
  python -c "
  import fiftyone.zoo as foz
  dataset = foz.load_zoo_dataset('open-images-v7', split='train',
      label_types=['detections'],
      classes=['Vehicle registration plate'],
      max_samples=5000)
  dataset.export(export_dir='datasets/openimages_plates/',
      dataset_type=foz.types.YOLOv5Dataset)
  "
  ```

---

#### B2. UFPR-ALPR (Federal Univ. Paraná — ALPR Dataset)

- **URL**: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/
- **Size**: 4,500 images, 30 video sequences
- **Annotations**: Plate bounding box + plate text ground truth
- **Why useful**: Specifically designed for ALPR (Automatic Licence Plate Recognition) — both detection and OCR ground truth provided.
- **Note**: Primarily Brazilian plates, but plate bounding box annotations transfer perfectly to training the plate localiser.

---

#### B3. CCPD (Chinese City Parking Dataset)

- **URL**: https://github.com/detectRecog/CCPD
- **Kaggle**: https://www.kaggle.com/datasets/nicholasjhana/ccpd-2019-chinese-city-parking
- **Size**: 290,000 images with bounding box + plate text
- **Why useful**: Extremely large dataset with highly diverse conditions (tilt, blur, night, rain, partial occlusion). Even though plates are Chinese, the plate localisation bounding box annotations are universally useful.
- **Recommended subset**: Download CCPD-Base (~100k) + CCPD-Blur (~20k) + CCPD-Night (~20k).

---

#### B4. Indian Number Plate Detection — Roboflow

- **URL**: https://universe.roboflow.com
- **Search**: `"indian number plate detection"` / `"ILP detection"` / `"vehicle registration plate india"`
- **Expected**: Multiple datasets, ~3,000–10,000 images total
- **Format**: Export in YOLOv8 format
- **Critical Note**: Always include Indian LP datasets. Western LP datasets have different plate proportions, fonts, and mounting positions. Indian HSRPs (High Security Registration Plates) have a specific blue strip on the left — train on this variant.

---

#### B5. Self-Collection at Deployment Site (MANDATORY — 200 minimum)

No dataset covers your exact camera angle, focal length, mounting height, and local conditions. **You must collect and annotate a minimum of 200 images from the actual deployment camera.**

```
Collection protocol:
1. Run camera for 2 hours during peak traffic (8–10 AM or 5–7 PM)
2. Extract frames with two-wheelers visible
3. Annotate plate bounding boxes using Roboflow Annotate (free tier)
4. Include night-time images: collect 50–100 frames after 7 PM
5. Include rain/wet conditions: collect 30–50 frames during rain
6. Export in YOLOv8 format, merge into training set
7. Split: add all 200 to training set (prioritise local data)
```

---

#### Dataset B — Final Combined Strategy

```
Total training images for plate localisation:
  Open Images V7 (plate class):            ~5,000  images
  CCPD-Base + CCPD-Blur + CCPD-Night:     ~20,000  images
  UFPR-ALPR:                               ~4,500  images
  Roboflow Indian LP sets:                 ~5,000  images
  Self-collected local deployment images:    ~200  images
  ─────────────────────────────────────────────────────
  TOTAL:                                  ~34,700  images

Split: 75% train / 15% val / 10% test
```

---

### Dataset Group C — OCR / Plate Text Recognition

---

#### C1. UFPR-ALPR OCR Labels (Same as B2)

- Already described above. Provides plate text ground truth — use for OCR model fine-tuning.

---

#### C2. Indian LP OCR Dataset — Kaggle

- **URL**: https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset
- **URL**: https://www.kaggle.com/datasets/dataclusterlabs/indian-license-plates-dataset
- **Size**: ~5,000–10,000 cropped plate images with text labels
- **Format**: Images + CSV with plate number ground truth
- **Critical For**: Fine-tuning PaddleOCR's recognition head on Indian LP fonts (specifically IND font, Cargo font, and the HSRP standard font).

---

#### C3. Synthetic Indian LP Dataset — Generate Your Own

For OCR fine-tuning, synthetic data is extremely effective and easy to generate. Use Python to render synthetic plates with the exact fonts used on Indian vehicles.

```python
# generate_synthetic_plates.py
from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import numpy as np
import cv2

STATES = ["KA", "MH", "DL", "TN", "UP", "GJ", "RJ", "WB", "AP", "TS",
          "KL", "PB", "HR", "MP", "CG", "BR", "OD", "JH", "UK", "HP"]

def random_plate_number() -> str:
    state = random.choice(STATES)
    district = f"{random.randint(1, 99):02d}"
    series = "".join(random.choices(string.ascii_uppercase, k=2))
    number = f"{random.randint(1000, 9999)}"
    return f"{state}{district}{series}{number}"

def render_plate(plate_text: str, width: int = 400,
                 height: int = 100) -> np.ndarray:
    """Render a synthetic Indian LP image."""
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to use a monospace font close to Indian LP font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 
                                   size=60)
    except:
        font = ImageFont.load_default()
    
    # Draw plate border
    draw.rectangle([(2, 2), (width-3, height-3)], 
                   outline=(0, 0, 0), width=3)
    
    # Draw text centered
    bbox = draw.textbbox((0, 0), plate_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), plate_text, fill=(0, 0, 0), font=font)
    
    img_np = np.array(img)
    
    # Add realistic degradation
    img_np = add_plate_degradation(img_np)
    
    return img_np

def add_plate_degradation(img: np.ndarray) -> np.ndarray:
    """Add realistic plate degradation for robust training."""
    aug_choice = random.randint(0, 5)
    
    if aug_choice == 0:  # Gaussian noise
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif aug_choice == 1:  # Slight blur (motion)
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, 1), 0)  # horizontal motion blur
    elif aug_choice == 2:  # Brightness variation
        factor = random.uniform(0.5, 1.4)
        img = np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)
    elif aug_choice == 3:  # Slight rotation
        angle = random.uniform(-5, 5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif aug_choice == 4:  # Grease/dirt overlay
        mask = np.random.random(img.shape[:2]) < 0.05
        img[mask] = np.random.randint(100, 200, size=img[mask].shape)
    # aug_choice == 5: no augmentation (clean plate)
    
    return img

# Generate 10,000 synthetic plates
os.makedirs("datasets/synthetic_plates/images", exist_ok=True)
labels = []

for i in range(10000):
    plate_text = random_plate_number()
    plate_img  = render_plate(plate_text)
    img_path   = f"datasets/synthetic_plates/images/{i:05d}.jpg"
    cv2.imwrite(img_path, plate_img)
    labels.append(f"{img_path}\t{plate_text}")

with open("datasets/synthetic_plates/labels.txt", "w") as f:
    f.write("\n".join(labels))

print(f"Generated 10,000 synthetic plates.")
```

---

#### Dataset C — Final Combined Strategy

```
Total training images for OCR fine-tuning:
  UFPR-ALPR plate crops:                   ~4,500  images
  Indian LP Kaggle datasets:               ~8,000  images
  Synthetic Indian plates (generated):    ~10,000  images
  Self-collected annotated plates:           ~200  images
  ─────────────────────────────────────────────────────
  TOTAL:                                  ~22,700  images
```

---

## 12. Model Training — Step-by-Step

### Environment Setup

```bash
# Create environment
conda create -n obj3_footpath python=3.10 -y
conda activate obj3_footpath

# Core dependencies
pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install paddlepaddle-gpu paddleocr
pip install opencv-python-headless albumentations
pip install paho-mqtt fastapi uvicorn  # for dashboard integration
pip install tflite-runtime             # for Pi deployment (not training machine)
pip install onnx onnxruntime-gpu paddle2onnx
pip install fiftyone roboflow pillow numpy scipy

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

### Training Step 1 — Two-Wheeler Detection (YOLOv8n)

```bash
# Prepare merged dataset structure
python scripts/merge_datasets.py \
    --sources datasets/coco_twowheelers \
               datasets/idd_twowheeler \
               datasets/bdd100k_twowheeler \
               datasets/roboflow_indian_twowheeler \
    --output  datasets/merged_twowheeler \
    --split 0.75 0.15 0.10

# Verify dataset
yolo data check cfg=datasets/merged_twowheeler/data.yaml

# ── data.yaml ──────────────────────────────────────────────
# path: datasets/merged_twowheeler
# train: images/train
# val:   images/val
# test:  images/test
# nc: 4
# names: ['motorcycle', 'bicycle', 'e-scooter', 'scooter']
# ───────────────────────────────────────────────────────────

# Train
yolo detect train \
    model=yolov8n.pt \
    data=datasets/merged_twowheeler/data.yaml \
    epochs=120 \
    imgsz=640 \
    batch=32 \
    device=0 \
    workers=8 \
    project=runs/obj3 \
    name=twowheeler_v1 \
    pretrained=True \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    weight_decay=0.0005 \
    warmup_epochs=3 \
    mosaic=1.0 \
    mixup=0.1 \
    degrees=5.0 \
    translate=0.1 \
    scale=0.5 \
    flipud=0.1 \
    fliplr=0.5 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4

# Target metrics:
# mAP50 (motorcycle):  > 0.82
# mAP50 (bicycle):     > 0.78
# mAP50 (scooter):     > 0.75
# mAP50:95 (overall):  > 0.55
```

---

### Training Step 2 — Licence Plate Localisation (YOLOv8n-LP)

```bash
# Prepare LP dataset
python scripts/merge_datasets.py \
    --sources datasets/openimages_plates \
               datasets/ccpd_subset \
               datasets/ufpr_alpr \
               datasets/roboflow_indian_lp \
    --output  datasets/merged_lp_localise \
    --split 0.75 0.15 0.10

# ── lp_localise.yaml ───────────────────────────────────────
# path: datasets/merged_lp_localise
# train: images/train
# val:   images/val
# nc: 1
# names: ['licence_plate']
# ───────────────────────────────────────────────────────────

# Train (longer — small object, needs more epochs)
yolo detect train \
    model=yolov8n.pt \
    data=datasets/merged_lp_localise/lp_localise.yaml \
    epochs=200 \
    imgsz=640 \
    batch=32 \
    device=0 \
    name=lp_localise_v1 \
    pretrained=True \
    optimizer=AdamW \
    lr0=0.001 \
    patience=30 \
    mosaic=1.0 \
    scale=0.6 \
    degrees=8.0

# Target metrics:
# mAP50 (licence_plate):  > 0.88
# mAP50:95:               > 0.65
# Recall @ conf=0.3:      > 0.92   ← prioritise recall over precision for plates
```

---

### Training Step 3 — PaddleOCR Fine-Tuning

```bash
# This step fine-tunes the recognition (text reading) component of PaddleOCR
# The detection component (finding text regions) is already good from the base model

# Prepare recognition dataset
# Format: one image per plate crop + label file
# labels.txt format: image_path\tplate_text
cat datasets/ufpr_alpr/ocr_labels.txt \
    datasets/indian_lp_kaggle/ocr_labels.txt \
    datasets/synthetic_plates/labels.txt \
    datasets/self_collected/ocr_labels.txt \
    > datasets/ocr_combined/labels.txt

# Split
python scripts/split_ocr_dataset.py \
    --labels datasets/ocr_combined/labels.txt \
    --train-ratio 0.85 \
    --val-ratio 0.15

# Fine-tune PaddleOCR recognition model
python tools/train.py \
    -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml \
    -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy \
       Train.dataset.label_file_list=["datasets/ocr_combined/train_labels.txt"] \
       Eval.dataset.label_file_list=["datasets/ocr_combined/val_labels.txt"] \
       Global.epoch_num=60 \
       Global.save_model_dir=./output/paddleocr_indian_rec/ \
       Train.loader.batch_size_per_card=128

# Target: Character-level accuracy > 94% on Indian LP val set
# Word-level accuracy (full plate correct) > 85%
```

---

## 13. Edge Conversion & Quantisation

### Step 1 — YOLOv8n Two-Wheeler → TFLite INT8

```python
from ultralytics import YOLO

# Two-wheeler detector
model = YOLO("runs/obj3/twowheeler_v1/weights/best.pt")
model.export(
    format="tflite",
    int8=True,
    data="datasets/merged_twowheeler/data.yaml",
    imgsz=320,        # 320 for Pi 4, use 640 for Jetson
    simplify=True,
    nms=True,         # bake NMS into TFLite graph for simplicity
)
# Output: best_int8.tflite (~3.2 MB)
# Move to: models/twowheeler_int8.tflite

# LP localiser
model_lp = YOLO("runs/obj3/lp_localise_v1/weights/best.pt")
model_lp.export(
    format="tflite",
    int8=True,
    data="datasets/merged_lp_localise/lp_localise.yaml",
    imgsz=320,
    simplify=True,
)
# Output: best_int8.tflite (~3.2 MB)
# Move to: models/lp_localise_int8.tflite
```

### Step 2 — Verify Quantisation Accuracy

```bash
# CRITICAL: Always validate INT8 model accuracy matches FP32 model
yolo detect val \
    model=models/twowheeler_int8.tflite \
    data=datasets/merged_twowheeler/data.yaml \
    split=test \
    device=cpu

# Acceptable degradation: mAP50 drop < 3% from FP32 baseline
# If degradation > 5%, use more calibration images during export
```

### Step 3 — PaddleOCR → ONNX (For Edge)

```bash
# Export fine-tuned PaddleOCR to ONNX for ONNX Runtime on Pi/Jetson
paddle2onnx \
    --model_dir output/paddleocr_indian_rec/ \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file models/paddleocr_rec_indian.onnx \
    --opset_version 11 \
    --input_shape_dict "x:[-1,3,48,-1]"

# Verify ONNX model
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('models/paddleocr_rec_indian.onnx')
dummy = np.zeros((1, 3, 48, 320), dtype=np.float32)
out = sess.run(None, {'x': dummy})
print('ONNX model loaded OK. Output shape:', out[0].shape)
"
```

### Step 4 — Jetson Nano TensorRT Conversion (Optional, High Impact)

```python
# On Jetson Nano ONLY — run this on the device
from ultralytics import YOLO

# Convert to TensorRT FP16 for 3× speedup on Jetson
model = YOLO("models/twowheeler_int8.tflite")  # or use .pt file
model.export(
    format="engine",   # TensorRT
    device=0,          # Jetson GPU
    half=True,         # FP16
    simplify=True,
    workspace=2,       # GB — adjust to available VRAM
)
# Expected: ~22ms per frame at 640px (vs ~55ms TFLite on Jetson)
```

### Final Model Sizes on Deployment Device

```
models/ directory on edge device:
  twowheeler_int8.tflite          3.2 MB   (or .engine for Jetson)
  lp_localise_int8.tflite         3.2 MB
  paddleocr_det.onnx              2.8 MB   (pre-trained, no fine-tuning needed)
  paddleocr_rec_indian.onnx       8.0 MB   (fine-tuned on Indian LP)
  paddleocr_cls.onnx              1.0 MB   (angle classification)
  ────────────────────────────────────────
  TOTAL:                         18.2 MB   (fits comfortably on any device)
```

---

## 14. Full Inference Loop Code

```python
"""
obj3_main.py — Objective 3: Footpath Violation Detection & Auto-Enforcement
Full production inference loop for Raspberry Pi 4 / Jetson Nano
"""

import cv2
import json
import time
import datetime
import numpy as np
import os
import threading
from collections import defaultdict, deque
from pathlib import Path

import tflite_runtime.interpreter as tflite
from paddleocr import PaddleOCR

# ── Configuration ──────────────────────────────────────────────────────────────
with open("config/footpath_roi.json") as f:
    CAM_CONFIG = json.load(f)

with open("config/speed_calibration.json") as f:
    SPEED_CONFIG = json.load(f)

ROI_POLYGON         = np.array(CAM_CONFIG["footpath_roi"], dtype=np.int32)
PIXELS_PER_METRE    = SPEED_CONFIG["pixels_per_metre"]
CAMERA_FPS          = SPEED_CONFIG["camera_fps"]
SPEED_THRESHOLD     = 5.0    # km/h — below this = parked, skip challan
CONF_THRESHOLD      = 0.45
OVERLAP_THRESHOLD   = 0.35   # 35% of vehicle must overlap with ROI
OCR_CONF_THRESHOLD  = 0.65   # minimum OCR confidence to generate challan
COOLDOWN_SECONDS    = 60     # same vehicle won't get 2 challans within 60s

# ── Model Loading ───────────────────────────────────────────────────────────────
print("[INIT] Loading models...")

from ultralytics import YOLO

# Use Ultralytics for detection + ByteTrack (handles both tasks)
detector = YOLO("models/twowheeler_int8.tflite", task="detect")
lp_model  = YOLO("models/lp_localise_int8.tflite", task="detect")

# PaddleOCR (loads ONNX models internally)
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    rec_model_dir='models/paddleocr_rec_indian_inference',
    det_model_dir='models/paddleocr_det',
    cls_model_dir='models/paddleocr_cls',
    use_gpu=False,      # set True on Jetson
    show_log=False,
)

print("[INIT] All models loaded.")

# ── State ───────────────────────────────────────────────────────────────────────
track_history     = defaultdict(lambda: deque(maxlen=15))
recent_challans   = {}   # track_id → last_challan_timestamp
frame_count       = 0
SKIP_FRAMES       = 2    # process 1 in every 3 frames

# ── Utility Functions ──────────────────────────────────────────────────────────
def get_speed(track_id, center, fps, ppm):
    track_history[track_id].append(center)
    if len(track_history[track_id]) < 4:
        return 0.0
    pts = list(track_history[track_id])[-4:]
    dists = [np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
             for i in range(1, len(pts))]
    avg_px_per_frame = np.mean(dists)
    return round(avg_px_per_frame / ppm * fps * 3.6, 1)  # km/h

def is_on_footpath(bbox):
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int(bbox[3])  # bottom center
    result = cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False)
    return result >= 0

def get_plate_from_vehicle(vehicle_crop):
    """Run plate localiser → enhance → OCR."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None, 0.0
    
    # Plate localisation
    lp_results = lp_model(vehicle_crop, conf=0.3, verbose=False)
    if not lp_results or len(lp_results[0].boxes) == 0:
        return None, 0.0
    
    # Get best plate box
    boxes = lp_results[0].boxes.xyxy.cpu().numpy()
    confs = lp_results[0].boxes.conf.cpu().numpy()
    best_idx = np.argmax(confs)
    x1, y1, x2, y2 = [int(v) for v in boxes[best_idx]]
    
    # Clamp to image bounds
    h, w = vehicle_crop.shape[:2]
    x1, y1 = max(0, x1-5), max(0, y1-5)
    x2, y2 = min(w, x2+5), min(h, y2+5)
    
    plate_crop = vehicle_crop[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return None, 0.0
    
    # Enhance
    plate_enhanced = enhance_plate_cpu(plate_crop)  # from Stage 5
    
    # OCR
    ocr_result = ocr_engine.ocr(plate_enhanced, cls=True)
    if not ocr_result or not ocr_result[0]:
        return None, 0.0
    
    text = "".join([line[1][0] for line in ocr_result[0]])
    conf = np.mean([line[1][1] for line in ocr_result[0]])
    
    cleaned = clean_plate_text(text)
    return cleaned, float(conf)

def enhance_plate_cpu(plate_img):
    """Classical enhancement pipeline."""
    target_w = max(400, plate_img.shape[1] * 3)
    scale = target_w / plate_img.shape[1]
    new_h = int(plate_img.shape[0] * scale)
    up = cv2.resize(plate_img, (target_w, new_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0,0), 1.5)
    sharp = cv2.addWeighted(eq, 1.8, blur, -0.8, 0)
    denoised = cv2.bilateralFilter(sharp, 5, 40, 40)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

def clean_plate_text(raw):
    import re
    text = raw.upper().replace(' ', '').replace('-', '').replace('.', '')
    return re.sub(r'[^A-Z0-9]', '', text)

def validate_plate(text):
    import re
    pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')
    bh_pattern = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$')
    return bool(pattern.match(text) or bh_pattern.match(text))

def in_cooldown(track_id):
    if track_id not in recent_challans:
        return False
    elapsed = time.time() - recent_challans[track_id]
    return elapsed < COOLDOWN_SECONDS

def save_and_alert(frame, vehicle_crop, plate_text, plate_conf,
                   vehicle_class, speed, track_id):
    """Save evidence package + push alert."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = Path(f"violations/{ts}_{plate_text}")
    folder.mkdir(parents=True, exist_ok=True)
    
    # Annotate and save full frame
    annotated = frame.copy()
    cv2.putText(annotated, f"VIOLATION: {plate_text} {speed}km/h",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(str(folder / "evidence_frame.jpg"), annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    if vehicle_crop is not None:
        cv2.imwrite(str(folder / "vehicle_crop.jpg"), vehicle_crop)
    
    # Save metadata
    record = {
        "violation_id": str(uuid.uuid4()) if 'uuid' in dir() else ts,
        "timestamp": datetime.datetime.now().isoformat(),
        "plate_number": plate_text,
        "plate_ocr_confidence": round(plate_conf, 3),
        "plate_valid_format": validate_plate(plate_text),
        "vehicle_class": vehicle_class,
        "speed_kmph": speed,
        "camera_id": CAM_CONFIG.get("camera_id", "CAM-001"),
        "location": CAM_CONFIG.get("location_name", "Unknown"),
        "gps_lat": CAM_CONFIG.get("gps_lat", 0.0),
        "gps_lng": CAM_CONFIG.get("gps_lng", 0.0),
        "violation_type": "FOOTPATH_ENCROACHMENT",
        "fine_inr": 500,
        "evidence_dir": str(folder),
    }
    
    with open(folder / "violation_metadata.json", "w") as f:
        json.dump(record, f, indent=2)
    
    print(f"[CHALLAN] {plate_text} | {speed} km/h | conf={plate_conf:.2f} | {folder}")
    
    # Non-blocking dashboard push
    threading.Thread(
        target=push_to_dashboard, args=(record,), daemon=True
    ).start()
    
    recent_challans[track_id] = time.time()

def push_to_dashboard(record):
    """Push violation alert to police dashboard via MQTT."""
    try:
        import paho.mqtt.publish as publish
        publish.single(
            "footpath/violations",
            payload=json.dumps(record),
            hostname="mqtt.dashboard.local",
            port=1883,
        )
    except Exception as e:
        pass  # offline — already saved locally


# ── Main Loop ──────────────────────────────────────────────────────────────────
def run():
    global frame_count
    
    cap = cv2.VideoCapture("rtsp://192.168.1.100/stream1")
    # Or for USB camera: cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise latency
    
    print("[START] Footpath violation detection running...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed. Retrying...")
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture("rtsp://192.168.1.100/stream1")
            continue
        
        frame_count += 1
        if frame_count % (SKIP_FRAMES + 1) != 0:
            continue  # skip frames to maintain throughput
        
        t0 = time.time()
        
        # ── Stage 1+3: Detection + Tracking ────────────────────────────
        track_results = detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            iou=0.50,
            verbose=False,
        )
        
        if track_results is None or len(track_results[0].boxes) == 0:
            continue  # nothing detected
        
        boxes  = track_results[0].boxes.xyxy.cpu().numpy()
        ids    = track_results[0].boxes.id
        clses  = track_results[0].boxes.cls.cpu().numpy()
        
        if ids is None:
            continue  # tracker not yet initialised
        
        ids = ids.cpu().numpy().astype(int)
        
        class_names = {0: "motorcycle", 1: "bicycle",
                       2: "e-scooter",  3: "scooter"}
        
        for i, (bbox, track_id, cls_id) in enumerate(zip(boxes, ids, clses)):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            center = ((x1+x2)//2, (y1+y2)//2)
            vehicle_class = class_names.get(int(cls_id), "two-wheeler")
            
            # ── Stage 2: ROI check ──────────────────────────────────────
            if not is_on_footpath([x1, y1, x2, y2]):
                continue
            
            # ── Stage 3: Speed estimation ───────────────────────────────
            speed = get_speed(track_id, center, CAMERA_FPS, PIXELS_PER_METRE)
            
            if speed < SPEED_THRESHOLD:
                continue  # parked vehicle — not a moving violation
            
            # ── Cooldown check ──────────────────────────────────────────
            if in_cooldown(track_id):
                continue
            
            # ── Stage 4+5+6: Plate pipeline ─────────────────────────────
            padding = 20
            veh_crop = frame[max(0,y1-padding):min(frame.shape[0],y2+padding),
                             max(0,x1-padding):min(frame.shape[1],x2+padding)]
            
            plate_text, plate_conf = get_plate_from_vehicle(veh_crop)
            
            # ── Stage 7: Evidence packaging ─────────────────────────────
            if plate_text and plate_conf >= OCR_CONF_THRESHOLD:
                if validate_plate(plate_text):
                    save_and_alert(
                        frame, veh_crop, plate_text, plate_conf,
                        vehicle_class, speed, track_id
                    )
                else:
                    # Log unvalidated reading for manual review
                    print(f"[WARN] Invalid plate format: '{plate_text}' "
                          f"(conf={plate_conf:.2f}) — logged for manual review")
                    # Still save evidence with MANUAL_REVIEW flag
                    with open("violations/manual_review_queue.jsonl", "a") as f:
                        f.write(json.dumps({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "raw_plate": plate_text,
                            "confidence": plate_conf,
                            "speed_kmph": speed,
                            "vehicle_class": vehicle_class,
                        }) + "\n")
        
        latency_ms = (time.time() - t0) * 1000
        if frame_count % 100 == 0:
            print(f"[PERF] Frame {frame_count} | Latency: {latency_ms:.1f}ms")
    
    cap.release()

if __name__ == "__main__":
    run()
```

---

## 15. Evaluation & Acceptance Criteria

### Per-Stage Acceptance Criteria

| Stage | Model | Metric | Minimum | Production Target |
|---|---|---|---|---|
| 1 — Two-wheeler detection | YOLOv8n INT8 | mAP50 (motorcycle) | > 0.80 | > 0.88 |
| 1 — Two-wheeler detection | YOLOv8n INT8 | mAP50 (scooter) | > 0.75 | > 0.83 |
| 1 — Two-wheeler detection | YOLOv8n INT8 | False Negative Rate | < 12% | < 6% |
| 1 — Two-wheeler detection | YOLOv8n INT8 | FPS on Pi 4 @ 320px | > 12 FPS | > 15 FPS |
| 4 — Plate localisation | YOLOv8n-LP INT8 | mAP50 (plate) | > 0.85 | > 0.92 |
| 4 — Plate localisation | YOLOv8n-LP INT8 | Recall @ conf=0.3 | > 0.90 | > 0.95 |
| 6 — OCR | PaddleOCR fine-tuned | Character accuracy | > 90% | > 95% |
| 6 — OCR | PaddleOCR fine-tuned | Word accuracy (full plate) | > 80% | > 90% |
| 6 — OCR | PaddleOCR fine-tuned | Valid format match rate | > 85% | > 93% |
| FULL PIPELINE | All stages | e-Challan precision | > 88% | > 95% |
| FULL PIPELINE | All stages | e-Challan recall | > 75% | > 85% |
| FULL PIPELINE | All stages | End-to-end latency (Pi 4) | < 250ms | < 180ms |
| FULL PIPELINE | All stages | End-to-end latency (Jetson) | < 100ms | < 70ms |

### Test Protocol

```
Test Set Composition (collect before training — never use for training):
  Normal daylight two-wheelers on footpath:      50 clips
  Night-time violations (IR illumination):       20 clips
  Rain/wet conditions:                           15 clips
  Partial occlusion (other pedestrians):         20 clips
  Fast-moving bike (> 30 km/h):                  15 clips
  Parked two-wheelers (should NOT trigger):       30 clips (false positive test)
  Pedestrians only (should NOT trigger):          30 clips (false positive test)
  ────────────────────────────────────────────────────────
  TOTAL:                                        180 clips

For each clip:
  1. Run full pipeline
  2. Record: detected (Y/N), plate read correctly (Y/N), challan generated (Y/N)
  3. Compare against ground-truth label

Key metrics to report:
  - Precision = correct challans / total challans generated
  - Recall    = correct challans / total actual violations
  - False Positive Rate = spurious challans / total non-violations
  - Night-time accuracy vs. day accuracy (should be < 10% drop)
```

### Common Failure Modes & Fixes

| Failure Mode | Symptom | Fix |
|---|---|---|
| Two-wheeler not detected from elevated angle | FNR > 20% on overhead shots | Add top-down view images from IDD + Roboflow during training |
| Plate unreadable due to motion blur | OCR confidence always < 0.5 | Switch to global shutter camera; add motion blur augmentation in OCR training |
| Duplicate challans for same vehicle | Multiple challans per crossing | Tune cooldown timer (increase to 120s); verify ByteTrack persistence |
| Pedestrian detected as bicycle | False positives on pedestrians | Increase CONF_THRESHOLD to 0.55; add hard negative mining (pedestrians in training) |
| Plates not localised at night | Plate localiser recall < 0.7 at night | Add IR images to LP training set; lower LP conf threshold to 0.25 at night |
| Invalid plate format after OCR | Validation rejects 30%+ of readings | Fine-tune PaddleOCR on more Indian LP data; add more synthetic plates |

---

## 16. Deployment Checklist

### Pre-Installation (Office/Lab)

- [ ] All models trained and validated against acceptance criteria
- [ ] All models converted to TFLite INT8 / ONNX
- [ ] Pipeline tested on recorded test video (not live camera)
- [ ] All false positive / false negative rates within spec
- [ ] `systemd` service configured and tested for auto-start
- [ ] SD card / SSD imaged with complete system (use `rpi-clone` for backup)

### On-Site Installation

- [ ] Camera mounted at recommended height: **3.5–5 metres** from ground, angled downward at 25–35°
- [ ] Camera positioned to see the full footpath width in frame
- [ ] ROI calibration tool run and polygon saved to `config/footpath_roi.json`
- [ ] Speed calibration run with 1-metre reference marker saved to `config/speed_calibration.json`
- [ ] IR illuminator aimed at footpath coverage zone
- [ ] Night-time test run: 10-minute recording reviewed for detection quality
- [ ] Network connectivity confirmed (LAN / 4G router)
- [ ] MQTT broker address configured in `config/dashboard.json`
- [ ] Test violation triggered manually (push bike through ROI) — verify challan generated
- [ ] Weatherproof enclosure sealed (IP65 or higher)

### Post-Deployment Monitoring (First Week)

- [ ] Review all generated challans daily for first 3 days
- [ ] Check for unexpected false positives (pedestrians, leaves, shadows)
- [ ] Monitor `violations/manual_review_queue.jsonl` for unvalidated plate readings
- [ ] Check system latency logs — verify no thermal throttling on Pi
- [ ] Verify systemd service auto-restarts after simulated power cut
- [ ] Confirm cloud sync of violation logs when network available

---

## 17. File & Folder Structure

```
objective_3_footpath/
│
├── main.py                        # Entry point — full inference loop
├── requirements_edge.txt          # Pi/Jetson runtime dependencies
├── requirements_training.txt      # Training machine dependencies
│
├── config/
│   ├── footpath_roi.json          # ROI polygon — set at installation
│   ├── speed_calibration.json     # Pixel-to-metre ratio
│   ├── violation_rules.json       # Speed threshold, cooldown, fines
│   └── dashboard.json             # MQTT broker, API endpoint config
│
├── models/
│   ├── twowheeler_int8.tflite     # Stage 1 — two-wheeler detector
│   ├── lp_localise_int8.tflite    # Stage 4 — plate localiser
│   ├── paddleocr_det/             # Stage 6 — PaddleOCR detector component
│   ├── paddleocr_rec_indian/      # Stage 6 — fine-tuned OCR recogniser
│   └── paddleocr_cls/             # Stage 6 — angle classifier
│
├── scripts/
│   ├── calibration_tool.py        # Interactive ROI + speed calibration
│   ├── merge_datasets.py          # Merge multiple YOLO datasets
│   ├── split_ocr_dataset.py       # Split OCR labels into train/val
│   ├── generate_synthetic_plates.py # Synthetic LP image generator
│   ├── benchmark_pipeline.py      # Measure end-to-end latency on device
│   └── evaluate_pipeline.py       # Compute precision/recall on test set
│
├── training/
│   ├── train_twowheeler.sh        # Training command for Stage 1 model
│   ├── train_lp_localise.sh       # Training command for Stage 4 model
│   ├── finetune_paddleocr.sh      # Fine-tuning command for Stage 6
│   └── export_models.py           # Convert all models to TFLite/ONNX
│
├── violations/                    # Auto-created at runtime
│   ├── 2025-01-15_14-23-07_KA05AB1234/
│   │   ├── evidence_frame.jpg
│   │   ├── vehicle_crop.jpg
│   │   └── violation_metadata.json
│   └── manual_review_queue.jsonl  # Unvalidated plate readings
│
└── logs/
    ├── pipeline_latency.log       # Per-frame latency log
    └── system.log                 # Errors, restarts, warnings
```

---

## 18. Development Timeline

| Phase | Week | Task | Deliverable | Acceptance Check |
|---|---|---|---|---|
| **Data** | Week 1 | Download COCO, IDD, BDD100K; convert to YOLO format | `datasets/merged_twowheeler/` | `yolo data check` passes |
| **Data** | Week 1–2 | Download CCPD, OpenImages, Roboflow LP sets; merge | `datasets/merged_lp_localise/` | Dataset verified |
| **Data** | Week 2 | Generate 10,000 synthetic LP images | `datasets/synthetic_plates/` | Visual quality check |
| **Data** | Week 2 | Collect 200+ images from deployment camera, annotate | `datasets/self_collected/` | 200 images labelled |
| **Train** | Week 3 | Train YOLOv8n two-wheeler detector (120 epochs) | `runs/obj3/twowheeler_v1/weights/best.pt` | mAP50 > 0.80 |
| **Train** | Week 3–4 | Train YOLOv8n plate localiser (200 epochs) | `runs/obj3/lp_localise_v1/weights/best.pt` | mAP50 > 0.85 |
| **Train** | Week 4–5 | Fine-tune PaddleOCR on Indian LP data (60 epochs) | `output/paddleocr_indian_rec/` | Word accuracy > 80% |
| **Convert** | Week 5 | Convert all models to TFLite INT8 + ONNX | `models/*.tflite`, `models/*.onnx` | Size < 20MB total |
| **Convert** | Week 5 | Benchmark on Pi 4 and Jetson; verify latency | Latency report | Pi4 < 250ms, Jetson < 100ms |
| **Integrate** | Week 6 | Write full inference loop `main.py` | `main.py` running on device | No crashes in 30-min run |
| **Integrate** | Week 6 | Implement ROI calibration tool | `scripts/calibration_tool.py` | ROI saved correctly |
| **Test** | Week 7 | Run test protocol (180 clips) | Evaluation report | Precision > 88%, Recall > 75% |
| **Test** | Week 7 | Night-time & adverse condition testing | Night test results | < 10% accuracy drop vs. day |
| **Deploy** | Week 8 | On-site installation + calibration | System live on pole | Test challan generated |
| **Monitor** | Week 8–9 | Daily review of first 200 challans | Manual review log | False positive rate < 5% |

---

## Quick Reference — Key Numbers

| Parameter | Value |
|---|---|
| Two-wheeler detection input size (Pi 4) | 320 × 320 px |
| Two-wheeler detection input size (Jetson) | 640 × 640 px |
| Minimum plate width for reliable OCR | 80 pixels |
| Speed threshold for moving violation | 5.0 km/h |
| Challan cooldown per vehicle (same track ID) | 60 seconds |
| ROI overlap threshold for violation trigger | 35% of vehicle bbox |
| OCR confidence minimum for auto-challan | 0.65 |
| Total model storage on device | ~18 MB |
| Pipeline FPS (detection only, Pi 4) | 12–15 FPS |
| Full pipeline end-to-end latency (Pi 4) | ~200–220 ms |
| Full pipeline end-to-end latency (Jetson) | ~80–100 ms |
| Recommended camera height | 3.5–5 metres |
| Recommended camera angle | 25–35° downward |
| IR illuminator range required | Minimum 10 metres |

---

*Objective 3 — Footpath Violation Detection & Auto-Enforcement*
*Edge AI System · YOLOv8n · ByteTrack · PaddleOCR PP-OCRv3 · TFLite INT8 · ONNX Runtime*
*Raspberry Pi 4 / Jetson Nano Deployment*

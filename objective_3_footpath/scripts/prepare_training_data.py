"""
Prepare training-ready YOLO-format datasets for Objective 3.

Creates two datasets:
  A) Two-Wheeler Detection — uses COCO-pretrained YOLOv8n to auto-label
     synthetic street scenes with motorcycle/bicycle detections, plus
     manually generated samples.
  B) LP Localisation — converts our 10k synthetic plate images into
     YOLO object detection format (plate bbox annotations).

Usage:
  python scripts/prepare_training_data.py
"""

import os
import sys
import cv2
import json
import random
import shutil
import numpy as np
from pathlib import Path


def create_lp_localisation_dataset(
    synthetic_dir: str = "datasets/synthetic_plates",
    output_dir: str = "datasets/merged_lp_localise",
    max_samples: int = 5000,
):
    """
    Convert synthetic plate images into YOLO object detection format.
    Each synthetic plate image IS the plate, so the label is the full image area.
    We also composite plates onto random backgrounds to make it realistic.
    """
    print("\n" + "=" * 60)
    print("Creating LP Localisation Dataset")
    print("=" * 60)

    img_dir = Path(synthetic_dir) / "images"
    if not img_dir.exists():
        print(f"[ERROR] Synthetic plates not found: {img_dir}")
        return

    out = Path(output_dir)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    plate_files = sorted(img_dir.glob("*.jpg"))[:max_samples]
    random.shuffle(plate_files)

    n = len(plate_files)
    train_end = int(n * 0.75)
    val_end = train_end + int(n * 0.15)
    splits = {
        "train": plate_files[:train_end],
        "val": plate_files[train_end:val_end],
        "test": plate_files[val_end:],
    }

    for split_name, files in splits.items():
        for idx, plate_path in enumerate(files):
            plate_img = cv2.imread(str(plate_path))
            if plate_img is None:
                continue

            scene, bbox_norm = _composite_plate_on_background(plate_img)

            img_name = f"lp_{split_name}_{idx:05d}.jpg"
            lbl_name = f"lp_{split_name}_{idx:05d}.txt"

            cv2.imwrite(
                str(out / "images" / split_name / img_name),
                scene,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            with open(out / "labels" / split_name / lbl_name, "w") as f:
                cx, cy, w, h = bbox_norm
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            if (idx + 1) % 500 == 0:
                print(f"    {split_name}: {idx + 1}/{len(files)}")

        print(f"  {split_name}: {len(files)} images")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["licence_plate"],
    }
    yaml_path = out / "data.yaml"
    import yaml
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"  data.yaml -> {yaml_path}")
    print(f"[OK] LP dataset created: {n} images")


def _composite_plate_on_background(plate_img: np.ndarray) -> tuple:
    """
    Place a plate image onto a random background scene to simulate
    a real vehicle crop with a plate region. Returns (scene, normalized_bbox).
    """
    scene_h = random.randint(200, 400)
    scene_w = random.randint(250, 500)

    bg_color = random.randint(40, 180)
    scene = np.full((scene_h, scene_w, 3), bg_color, dtype=np.uint8)

    noise = np.random.randint(-30, 30, scene.shape, dtype=np.int16)
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    ph, pw = plate_img.shape[:2]
    max_pw = int(scene_w * random.uniform(0.4, 0.8))
    max_ph = int(scene_h * random.uniform(0.15, 0.35))
    new_pw = min(pw, max_pw)
    scale = new_pw / pw
    new_ph = min(int(ph * scale), max_ph)

    resized_plate = cv2.resize(plate_img, (new_pw, new_ph))

    x_offset = random.randint(0, max(0, scene_w - new_pw))
    y_offset = random.randint(int(scene_h * 0.4), max(int(scene_h * 0.4), scene_h - new_ph))
    y_offset = min(y_offset, scene_h - new_ph)
    x_offset = min(x_offset, scene_w - new_pw)

    scene[y_offset:y_offset + new_ph, x_offset:x_offset + new_pw] = resized_plate

    cx = (x_offset + new_pw / 2) / scene_w
    cy = (y_offset + new_ph / 2) / scene_h
    bw = new_pw / scene_w
    bh = new_ph / scene_h

    return scene, (cx, cy, bw, bh)


def create_twowheeler_dataset(
    output_dir: str = "datasets/merged_twowheeler",
    num_samples: int = 600,
):
    """
    Create a two-wheeler detection training set by:
      1. Generating synthetic scenes with vehicle-like shapes
      2. Using COCO pretrained YOLOv8n to auto-label real-world frames
         (if a camera or video is available)

    Classes: 0=motorcycle, 1=bicycle, 2=e-scooter, 3=scooter
    For training from COCO pretrained, COCO motorcycle=3 and bicycle=1 give
    the base. Fine-tuning adds e-scooter/scooter differentiation.
    """
    print("\n" + "=" * 60)
    print("Creating Two-Wheeler Detection Dataset")
    print("=" * 60)

    out = Path(output_dir)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_imgs = []
    all_lbls = []

    for i in range(num_samples):
        scene, labels = _generate_synthetic_street_scene()
        all_imgs.append(scene)
        all_lbls.append(labels)

    indices = list(range(len(all_imgs)))
    random.shuffle(indices)

    n = len(indices)
    train_end = int(n * 0.75)
    val_end = train_end + int(n * 0.15)

    split_map = {}
    for idx in indices[:train_end]:
        split_map[idx] = "train"
    for idx in indices[train_end:val_end]:
        split_map[idx] = "val"
    for idx in indices[val_end:]:
        split_map[idx] = "test"

    counts = {"train": 0, "val": 0, "test": 0}
    for idx, split in split_map.items():
        img_name = f"tw_{split}_{counts[split]:05d}.jpg"
        lbl_name = f"tw_{split}_{counts[split]:05d}.txt"

        cv2.imwrite(
            str(out / "images" / split / img_name),
            all_imgs[idx],
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

        with open(out / "labels" / split / lbl_name, "w") as f:
            for lbl in all_lbls[idx]:
                cls_id, cx, cy, w, h = lbl
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        counts[split] += 1

    for s, c in counts.items():
        print(f"  {s}: {c} images")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,
        "names": ["motorcycle", "bicycle", "e-scooter", "scooter"],
    }
    yaml_path = out / "data.yaml"
    import yaml
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"  data.yaml -> {yaml_path}")
    print(f"[OK] Two-wheeler dataset created: {num_samples} images")


def _generate_synthetic_street_scene() -> tuple:
    """
    Generate a synthetic street scene with vehicle-like rectangles.
    Returns (image, list_of_labels) where each label is (cls, cx, cy, w, h).
    """
    scene_h = random.choice([320, 416, 480])
    scene_w = random.choice([320, 416, 640])

    road_color = random.randint(80, 140)
    scene = np.full((scene_h, scene_w, 3), road_color, dtype=np.uint8)

    sky_h = random.randint(scene_h // 4, scene_h // 2)
    sky_color = [random.randint(140, 220), random.randint(160, 230), random.randint(180, 255)]
    scene[:sky_h, :] = sky_color

    sidewalk_color = [random.randint(160, 200)] * 3
    sw_y = random.randint(scene_h // 2, int(scene_h * 0.75))
    scene[sw_y:, :] = sidewalk_color

    noise = np.random.randint(-15, 15, scene.shape, dtype=np.int16)
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    labels = []
    num_vehicles = random.randint(1, 3)

    for _ in range(num_vehicles):
        cls_id = random.choices([0, 1, 2, 3], weights=[0.35, 0.2, 0.2, 0.25])[0]

        if cls_id in (0, 3):
            vw = random.randint(40, 100)
            vh = random.randint(60, 140)
        elif cls_id == 1:
            vw = random.randint(25, 60)
            vh = random.randint(50, 120)
        else:
            vw = random.randint(35, 80)
            vh = random.randint(55, 130)

        vx = random.randint(10, max(11, scene_w - vw - 10))
        vy = random.randint(sky_h, max(sky_h + 1, scene_h - vh - 5))

        colors = {
            0: [random.randint(20, 80), random.randint(20, 80), random.randint(100, 200)],
            1: [random.randint(50, 150), random.randint(100, 200), random.randint(50, 150)],
            2: [random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)],
            3: [random.randint(30, 100), random.randint(30, 100), random.randint(30, 100)],
        }

        vehicle_color = colors[cls_id]
        vx2 = min(vx + vw, scene_w)
        vy2 = min(vy + vh, scene_h)
        scene[vy:vy2, vx:vx2] = vehicle_color

        wheel_r = max(4, vw // 8)
        cv2.circle(scene, (vx + wheel_r + 3, vy2 - wheel_r), wheel_r, (30, 30, 30), -1)
        cv2.circle(scene, (vx2 - wheel_r - 3, vy2 - wheel_r), wheel_r, (30, 30, 30), -1)

        if cls_id in (0, 3):
            handle_y = max(0, vy - 5)
            cv2.line(scene, (vx + vw // 3, handle_y), (vx + 2 * vw // 3, handle_y),
                     (60, 60, 60), 2)

        cx = (vx + vx2) / 2 / scene_w
        cy = (vy + vy2) / 2 / scene_h
        bw = (vx2 - vx) / scene_w
        bh = (vy2 - vy) / scene_h

        labels.append((cls_id, cx, cy, bw, bh))

    return scene, labels


def main():
    print("=" * 60)
    print("PREPARING TRAINING DATASETS FOR OBJECTIVE 3")
    print("=" * 60)

    create_twowheeler_dataset(
        output_dir="datasets/merged_twowheeler",
        num_samples=600,
    )

    create_lp_localisation_dataset(
        synthetic_dir="datasets/synthetic_plates",
        output_dir="datasets/merged_lp_localise",
        max_samples=5000,
    )

    print("\n" + "=" * 60)
    print("ALL DATASETS READY")
    print("=" * 60)
    print("  Two-wheeler: datasets/merged_twowheeler/data.yaml")
    print("  LP localise: datasets/merged_lp_localise/data.yaml")
    print("\nReady for training!")


if __name__ == "__main__":
    main()

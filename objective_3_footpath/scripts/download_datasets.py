"""
Download and prepare all datasets required for Objective 3.

Dataset Groups:
  A — Two-Wheeler Detection (COCO, IDD, BDD100K, Roboflow)
  B — Licence Plate Localisation (Open Images, CCPD, UFPR, Roboflow)
  C — OCR Fine-Tuning (UFPR, Indian LP Kaggle, Synthetic)

Usage:
  python scripts/download_datasets.py --group all
  python scripts/download_datasets.py --group A
  python scripts/download_datasets.py --group B
  python scripts/download_datasets.py --group C
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DATASETS_ROOT = Path("datasets")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_coco_twowheelers():
    """Download COCO 2017 motorcycle/bicycle subset via fiftyone."""
    out_dir = DATASETS_ROOT / "coco_twowheelers"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[SKIP] {out_dir} already exists.")
        return

    print("[DOWNLOAD] COCO 2017 — motorcycle + bicycle subset via fiftyone...")
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz

        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            label_types=["detections"],
            classes=["motorcycle", "bicycle"],
            max_samples=8000,
        )
        dataset.export(
            export_dir=str(out_dir),
            dataset_type=fo.types.YOLOv5Dataset,
        )
        print(f"[OK] COCO two-wheeler subset → {out_dir}")
    except ImportError:
        print("[WARN] fiftyone not installed. Install with: pip install fiftyone")
    except Exception as e:
        print(f"[ERROR] COCO download failed: {e}")


def download_openimages_plates():
    """Download Open Images V7 licence plate class via fiftyone."""
    out_dir = DATASETS_ROOT / "openimages_plates"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[SKIP] {out_dir} already exists.")
        return

    print("[DOWNLOAD] Open Images V7 — Vehicle registration plate...")
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz

        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=["Vehicle registration plate"],
            max_samples=5000,
        )
        dataset.export(
            export_dir=str(out_dir),
            dataset_type=fo.types.YOLOv5Dataset,
        )
        print(f"[OK] Open Images plates → {out_dir}")
    except ImportError:
        print("[WARN] fiftyone not installed.")
    except Exception as e:
        print(f"[ERROR] Open Images download failed: {e}")


def download_roboflow_twowheelers():
    """Download Indian two-wheeler datasets from Roboflow."""
    out_dir = DATASETS_ROOT / "roboflow_indian_twowheeler"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[SKIP] {out_dir} already exists.")
        return

    print("[INFO] Roboflow Indian two-wheeler dataset:")
    print("  1. Go to https://universe.roboflow.com")
    print('  2. Search: "two wheeler detection india" or "motorcycle detection"')
    print("  3. Export in YOLOv8 format")
    print(f"  4. Place files in: {out_dir}")
    print()
    print("  Alternatively, use roboflow API:")
    print("  from roboflow import Roboflow")
    print("  rf = Roboflow(api_key='YOUR_KEY')")
    print("  project = rf.workspace().project('two-wheeler-detection')")
    print("  dataset = project.version(1).download('yolov8')")
    ensure_dir(out_dir)


def download_roboflow_indian_lp():
    """Download Indian LP datasets from Roboflow."""
    out_dir = DATASETS_ROOT / "roboflow_indian_lp"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[SKIP] {out_dir} already exists.")
        return

    print("[INFO] Roboflow Indian LP dataset:")
    print("  1. Go to https://universe.roboflow.com")
    print('  2. Search: "indian number plate detection"')
    print("  3. Export in YOLOv8 format")
    print(f"  4. Place files in: {out_dir}")
    ensure_dir(out_dir)


def download_kaggle_indian_lp():
    """Download Indian LP OCR dataset from Kaggle."""
    out_dir = DATASETS_ROOT / "indian_lp_kaggle"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[SKIP] {out_dir} already exists.")
        return

    print("[DOWNLOAD] Kaggle Indian Vehicle Dataset...")
    ensure_dir(out_dir)
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "saisirishan/indian-vehicle-dataset",
                "-p", str(out_dir), "--unzip",
            ],
            check=True,
        )
        print(f"[OK] Indian LP Kaggle → {out_dir}")
    except FileNotFoundError:
        print("[WARN] kaggle CLI not found. Install: pip install kaggle")
        print("  Then set up ~/.kaggle/kaggle.json with your API credentials.")
    except Exception as e:
        print(f"[ERROR] Kaggle download failed: {e}")


def download_idd():
    """Instructions for Indian Driving Dataset (requires manual registration)."""
    print("[INFO] Indian Driving Dataset (IDD):")
    print("  IDD requires free registration at https://idd.insaan.iiit.ac.in/")
    print("  1. Register and download IDD Detection dataset")
    print("  2. Extract to datasets/idd_twowheeler/")
    print("  3. Convert PASCAL VOC XML labels to YOLO format using merge_datasets.py")
    print("  This is the MOST IMPORTANT external dataset for Indian vehicle detection.")
    ensure_dir(DATASETS_ROOT / "idd_twowheeler")


def download_ccpd():
    """Download CCPD licence plate dataset from Kaggle."""
    out_dir = DATASETS_ROOT / "ccpd_subset"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[SKIP] {out_dir} already exists.")
        return

    print("[DOWNLOAD] CCPD Chinese plate dataset (for LP localisation bboxes)...")
    ensure_dir(out_dir)
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "nicholasjhana/ccpd-2019-chinese-city-parking",
                "-p", str(out_dir), "--unzip",
            ],
            check=True,
        )
        print(f"[OK] CCPD → {out_dir}")
    except FileNotFoundError:
        print("[WARN] kaggle CLI not found.")
    except Exception as e:
        print(f"[ERROR] CCPD download failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download Objective 3 datasets")
    parser.add_argument(
        "--group",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Which dataset group to download",
    )
    args = parser.parse_args()

    ensure_dir(DATASETS_ROOT)

    if args.group in ("A", "all"):
        print("\n" + "=" * 60)
        print("GROUP A — Two-Wheeler Detection Datasets")
        print("=" * 60)
        download_coco_twowheelers()
        download_idd()
        download_roboflow_twowheelers()

    if args.group in ("B", "all"):
        print("\n" + "=" * 60)
        print("GROUP B — Licence Plate Localisation Datasets")
        print("=" * 60)
        download_openimages_plates()
        download_ccpd()
        download_roboflow_indian_lp()

    if args.group in ("C", "all"):
        print("\n" + "=" * 60)
        print("GROUP C — OCR Fine-Tuning Datasets")
        print("=" * 60)
        download_kaggle_indian_lp()
        print("\n[INFO] Run generate_synthetic_plates.py to create 10k synthetic plates.")

    print("\n[DONE] Dataset download process complete.")
    print("Next steps:")
    print("  1. Manually download IDD from https://idd.insaan.iiit.ac.in/")
    print("  2. Run: python scripts/merge_datasets.py")
    print("  3. Run: python scripts/generate_synthetic_plates.py")


if __name__ == "__main__":
    main()

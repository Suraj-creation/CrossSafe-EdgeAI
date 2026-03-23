"""
Training Step 1 — YOLOv8n Two-Wheeler Detection Model.

Fine-tunes YOLOv8n on merged two-wheeler dataset (COCO + IDD + BDD100K + Roboflow).
Classes: motorcycle, bicycle, e-scooter, scooter

Target metrics:
  mAP50 (motorcycle) > 0.82
  mAP50 (bicycle)    > 0.78
  mAP50 (scooter)    > 0.75

Usage:
  python training/train_twowheeler.py
  python training/train_twowheeler.py --epochs 150 --batch 16 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(args):
    print("=" * 60)
    print("OBJECTIVE 3 — Stage 1: Two-Wheeler Detection Training")
    print("=" * 60)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        print("Run scripts/merge_datasets.py first to create the merged dataset.")
        return

    model = YOLO(args.model)
    print(f"[INFO] Base model: {args.model}")
    print(f"[INFO] Dataset:    {data_yaml}")
    print(f"[INFO] Epochs:     {args.epochs}")
    print(f"[INFO] Image size: {args.imgsz}")
    print(f"[INFO] Batch size: {args.batch}")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project="runs/obj3",
        name="twowheeler_v1",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=1.0,
        mixup=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        patience=25,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best weights: runs/obj3/twowheeler_v1/weights/best.pt")
    print(f"Last weights: runs/obj3/twowheeler_v1/weights/last.pt")

    print("\n[EVAL] Running validation on test split...")
    best = YOLO("runs/obj3/twowheeler_v1/weights/best.pt")
    metrics = best.val(data=str(data_yaml), split="test")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n Two-Wheeler Detector")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--data", default="datasets/merged_twowheeler/data.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

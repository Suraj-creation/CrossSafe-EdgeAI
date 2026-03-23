"""
Training Step 2 — YOLOv8n Licence Plate Localisation Model.

Fine-tunes YOLOv8n on merged LP dataset (Open Images + CCPD + UFPR + Indian LP).
Single class: licence_plate

Target metrics:
  mAP50 (licence_plate) > 0.88
  Recall @ conf=0.3     > 0.92

Usage:
  python training/train_lp_localiser.py
  python training/train_lp_localiser.py --epochs 200 --batch 32
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(args):
    print("=" * 60)
    print("OBJECTIVE 3 — Stage 4: Licence Plate Localiser Training")
    print("=" * 60)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        print("Run scripts/merge_datasets.py first with LP datasets.")
        return

    model = YOLO(args.model)
    print(f"[INFO] Base model: {args.model}")
    print(f"[INFO] Dataset:    {data_yaml}")
    print(f"[INFO] Epochs:     {args.epochs}")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project="runs/obj3",
        name="lp_localise_v1",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=30,
        mosaic=1.0,
        scale=0.6,
        degrees=8.0,
        translate=0.1,
        fliplr=0.0,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("LP LOCALISER TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best weights: runs/obj3/lp_localise_v1/weights/best.pt")

    print("\n[EVAL] Running validation on test split...")
    best = YOLO("runs/obj3/lp_localise_v1/weights/best.pt")
    metrics = best.val(data=str(data_yaml), split="test")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n LP Localiser")
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--data", default="datasets/merged_lp_localise/data.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

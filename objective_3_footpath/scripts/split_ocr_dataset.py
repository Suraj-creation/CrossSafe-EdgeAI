"""
Split OCR label files into train/val splits for PaddleOCR fine-tuning.

Input:  A combined labels.txt with lines: image_path\\tplate_text
Output: train_labels.txt and val_labels.txt

Usage:
  python scripts/split_ocr_dataset.py \
    --labels datasets/ocr_combined/labels.txt \
    --train-ratio 0.85 --val-ratio 0.15
"""

import argparse
import random
from pathlib import Path


def split_labels(labels_path: str, train_ratio: float, val_ratio: float):
    with open(labels_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and "\t" in l]

    print(f"Total OCR samples: {len(lines)}")
    random.shuffle(lines)

    n = len(lines)
    train_end = int(n * train_ratio)

    train_lines = lines[:train_end]
    val_lines = lines[train_end:]

    out_dir = Path(labels_path).parent
    train_path = out_dir / "train_labels.txt"
    val_path = out_dir / "val_labels.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines))

    print(f"  Train: {len(train_lines)} samples → {train_path}")
    print(f"  Val:   {len(val_lines)} samples → {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Split OCR labels into train/val")
    parser.add_argument("--labels", required=True, help="Combined labels.txt path")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    split_labels(args.labels, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()

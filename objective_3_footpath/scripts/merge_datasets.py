"""
Merge multiple YOLO-format datasets into a single unified dataset
with proper train/val/test splits.

Usage:
  python scripts/merge_datasets.py \
    --sources datasets/coco_twowheelers datasets/idd_twowheeler \
    --output datasets/merged_twowheeler \
    --split 0.75 0.15 0.10

  python scripts/merge_datasets.py \
    --sources datasets/openimages_plates datasets/ccpd_subset \
    --output datasets/merged_lp_localise \
    --split 0.75 0.15 0.10
"""

import argparse
import os
import random
import shutil
import yaml
from pathlib import Path
from collections import Counter


def find_image_label_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """Find all image-label pairs in a YOLO dataset directory."""
    pairs = []
    img_dirs = [
        source_dir / "images" / "train",
        source_dir / "images" / "val",
        source_dir / "images" / "test",
        source_dir / "images",
        source_dir / "train" / "images",
        source_dir / "valid" / "images",
        source_dir / "test" / "images",
        source_dir,
    ]
    label_dirs = [
        source_dir / "labels" / "train",
        source_dir / "labels" / "val",
        source_dir / "labels" / "test",
        source_dir / "labels",
        source_dir / "train" / "labels",
        source_dir / "valid" / "labels",
        source_dir / "test" / "labels",
    ]

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_dir in img_dirs:
        if not img_dir.is_dir():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in img_exts:
                continue
            label_name = img_path.stem + ".txt"
            for lbl_dir in label_dirs:
                lbl_path = lbl_dir / label_name
                if lbl_path.exists():
                    pairs.append((img_path, lbl_path))
                    break

    return pairs


def merge_datasets(
    sources: list[str],
    output_dir: str,
    split_ratios: tuple[float, float, float],
    class_names: list[str],
    dataset_name: str = "merged_dataset",
):
    output = Path(output_dir)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            print(f"[WARN] Source not found: {src}")
            continue
        pairs = find_image_label_pairs(src_path)
        print(f"  Found {len(pairs)} image-label pairs in {src}")
        all_pairs.extend(pairs)

    print(f"\nTotal pairs found: {len(all_pairs)}")

    if not all_pairs:
        print("[ERROR] No image-label pairs found. Check source directories.")
        return

    random.shuffle(all_pairs)

    n = len(all_pairs)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])

    splits = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:],
    }

    for split_name, pairs in splits.items():
        for idx, (img_path, lbl_path) in enumerate(pairs):
            new_name = f"{split_name}_{idx:06d}"
            new_img = output / "images" / split_name / f"{new_name}{img_path.suffix}"
            new_lbl = output / "labels" / split_name / f"{new_name}.txt"
            shutil.copy2(img_path, new_img)
            shutil.copy2(lbl_path, new_lbl)

        print(f"  {split_name}: {len(pairs)} samples")

    data_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = output / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\n[OK] Merged dataset saved to: {output}")
    print(f"     data.yaml: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge YOLO datasets")
    parser.add_argument("--sources", nargs="+", required=True, help="Source dataset directories")
    parser.add_argument("--output", required=True, help="Output merged dataset directory")
    parser.add_argument("--split", nargs=3, type=float, default=[0.75, 0.15, 0.10],
                        help="Train/val/test split ratios (default: 0.75 0.15 0.10)")
    parser.add_argument("--classes", nargs="+",
                        default=["motorcycle", "bicycle", "e-scooter", "scooter"],
                        help="Class names for data.yaml")
    parser.add_argument("--name", default="merged_dataset", help="Dataset name")
    args = parser.parse_args()

    merge_datasets(args.sources, args.output, tuple(args.split), args.classes, args.name)


if __name__ == "__main__":
    main()

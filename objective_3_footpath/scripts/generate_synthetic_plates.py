"""
Generate synthetic Indian licence plate images for OCR fine-tuning.

Generates 10,000 plates by default with realistic degradation:
  - Gaussian noise, motion blur, brightness variation
  - Slight rotation, dirt/grease overlay
  - Multiple Indian state codes and registration formats

Output:
  datasets/synthetic_plates/images/00000.jpg ... 09999.jpg
  datasets/synthetic_plates/labels.txt   (image_path\tplate_text)

Usage:
  python scripts/generate_synthetic_plates.py --count 10000
"""

import argparse
import os
import random
import string

import cv2
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("[ERROR] Pillow is required: pip install Pillow")
    raise

STATES = [
    "KA", "MH", "DL", "TN", "UP", "GJ", "RJ", "WB", "AP", "TS",
    "KL", "PB", "HR", "MP", "CG", "BR", "OD", "JH", "UK", "HP",
    "GA", "CH", "JK", "AS", "NL", "MN", "MZ", "TR", "SK", "AR",
]


def random_plate_standard() -> str:
    """Standard Indian LP: XX00XX0000"""
    state = random.choice(STATES)
    district = f"{random.randint(1, 99):02d}"
    series = "".join(random.choices(string.ascii_uppercase, k=random.choice([1, 2])))
    number = f"{random.randint(1000, 9999)}"
    return f"{state}{district}{series}{number}"


def random_plate_bh() -> str:
    """BH (Bharat) series: 00BH0000XX"""
    year = f"{random.randint(20, 26):02d}"
    number = f"{random.randint(1000, 9999)}"
    suffix = "".join(random.choices(string.ascii_uppercase, k=2))
    return f"{year}BH{number}{suffix}"


def random_plate_number() -> str:
    if random.random() < 0.1:
        return random_plate_bh()
    return random_plate_standard()


def find_font() -> ImageFont.FreeTypeFont:
    """Try to find a suitable monospace font."""
    font_candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/lucon.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/System/Library/Fonts/Courier.dfont",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size=56)
    return ImageFont.load_default()


def render_plate(plate_text: str, font: ImageFont.FreeTypeFont,
                 width: int = 400, height: int = 100) -> np.ndarray:
    bg_color = random.choice([
        (255, 255, 255),
        (245, 245, 220),
        (255, 255, 230),
        (240, 240, 240),
    ])
    text_color = (0, 0, 0)

    if random.random() < 0.15:
        bg_color = (255, 204, 0)
        text_color = (0, 0, 0)

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    draw.rectangle([(2, 2), (width - 3, height - 3)], outline=(0, 0, 0), width=3)

    if random.random() < 0.4:
        draw.rectangle([(3, 3), (40, height - 3)], fill=(0, 51, 153))
        draw.text((8, height // 2 - 8), "IND", fill=(255, 255, 255),
                  font=ImageFont.load_default())

    spaced = "  ".join(
        [plate_text[:4], plate_text[4:6], plate_text[6:]]
    ) if len(plate_text) >= 8 else plate_text

    bbox = draw.textbbox((0, 0), spaced, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), spaced, fill=text_color, font=font)

    return np.array(img)


def add_degradation(img: np.ndarray) -> np.ndarray:
    """Apply random realistic degradation to a plate image."""
    aug = random.randint(0, 7)

    if aug == 0:
        noise = np.random.normal(0, random.uniform(8, 25), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif aug == 1:
        k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, 1), 0)
    elif aug == 2:
        factor = random.uniform(0.4, 1.5)
        img = np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)
    elif aug == 3:
        angle = random.uniform(-7, 7)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif aug == 4:
        mask = np.random.random(img.shape[:2]) < random.uniform(0.02, 0.08)
        img[mask] = np.random.randint(80, 200, size=img[mask].shape).astype(np.uint8)
    elif aug == 5:
        k = random.choice([3, 5])
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)
    elif aug == 6:
        h, w = img.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx, dy = random.randint(-8, 8), random.randint(-5, 5)
        pts2 = np.float32([[dx, dy], [w - dx, dy], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Indian LP images")
    parser.add_argument("--count", type=int, default=10000, help="Number of plates to generate")
    parser.add_argument("--output", default="datasets/synthetic_plates", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    font = find_font()
    labels = []

    for i in range(args.count):
        plate_text = random_plate_number()
        plate_img = render_plate(plate_text, font)

        n_augs = random.randint(0, 2)
        for _ in range(n_augs):
            plate_img = add_degradation(plate_img)

        img_path = img_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(img_path), plate_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        labels.append(f"{img_path}\t{plate_text}")

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{args.count} plates...")

    labels_path = out / "labels.txt"
    with open(labels_path, "w") as f:
        f.write("\n".join(labels))

    print(f"\n[OK] Generated {args.count} synthetic plates.")
    print(f"     Images: {img_dir}")
    print(f"     Labels: {labels_path}")


if __name__ == "__main__":
    main()

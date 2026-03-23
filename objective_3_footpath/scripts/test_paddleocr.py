"""Test PaddleOCR on synthetic plates."""
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paddleocr import PaddleOCR

print("Initializing PaddleOCR 2.10...")
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
print("PaddleOCR initialized!")

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "datasets", "synthetic_plates", "images")

for i in [0, 1, 2, 3, 10, 50, 100]:
    p = os.path.join(base, f"{i:05d}.jpg")
    if os.path.exists(p):
        img = cv2.imread(p)
        big = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        r = ocr.ocr(big, cls=True)
        if r and r[0]:
            txt = r[0][0][1][0]
            conf = r[0][0][1][1]
            print(f"  Plate {i:05d}: '{txt}' (conf: {conf:.3f})")
        else:
            print(f"  Plate {i:05d}: No text detected")

print()
print("PaddleOCR is fully working!")

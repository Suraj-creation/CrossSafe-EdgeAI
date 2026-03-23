"""
Stage 5 — Plate Image Enhancement (Classical OpenCV, CPU-only).

Indian licence plates captured at distance, in motion, or at night are often
low-res, blurred, and poorly contrasted. This module enhances the plate crop
before OCR using CLAHE + Unsharp Mask + Bilateral Denoise.
Runs in ~8ms on Raspberry Pi 4.
"""

import cv2
import numpy as np


class PlateEnhancer:
    def __init__(
        self,
        target_width: int = 400,
        clahe_clip: float = 3.0,
        clahe_grid: tuple[int, int] = (4, 4),
        sharpen_weight: float = 1.8,
        blur_weight: float = -0.8,
        bilateral_d: int = 5,
        bilateral_sigma_color: int = 40,
        bilateral_sigma_space: int = 40,
    ):
        self.target_width = target_width
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=clahe_grid
        )
        self.sharpen_weight = sharpen_weight
        self.blur_weight = blur_weight
        self.bilateral_d = bilateral_d
        self.bilateral_sc = bilateral_sigma_color
        self.bilateral_ss = bilateral_sigma_space

    def enhance(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Full classical plate enhancement pipeline.
        Input:  BGR plate crop (any size)
        Output: Enhanced BGR image scaled to target_width
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img

        h, w = plate_img.shape[:2]
        target_w = max(self.target_width, w * 3)
        scale = target_w / w
        new_h = int(h * scale)

        upscaled = cv2.resize(
            plate_img, (target_w, new_h), interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        enhanced = self.clahe.apply(gray)

        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)

        sharpened = cv2.addWeighted(
            enhanced, self.sharpen_weight, blurred, self.blur_weight, 0
        )

        denoised = cv2.bilateralFilter(
            sharpened,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sc,
            sigmaSpace=self.bilateral_ss,
        )

        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    def deskew(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Correct slight rotation/tilt of plate using Hough line detection.
        Improves OCR accuracy on tilted plates.
        """
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=60)

        if lines is None:
            return plate_img

        angles = []
        for line in lines[:5]:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if abs(angle) < 20:
                angles.append(angle)

        if not angles:
            return plate_img

        median_angle = float(np.median(angles))
        if abs(median_angle) < 1.0:
            return plate_img

        h, w = plate_img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        deskewed = cv2.warpAffine(
            plate_img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return deskewed

    def full_pipeline(self, plate_img: np.ndarray) -> np.ndarray:
        """Run deskew then enhance — the recommended full path."""
        deskewed = self.deskew(plate_img)
        return self.enhance(deskewed)

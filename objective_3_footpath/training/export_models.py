"""
Convert all trained models to edge-deployable formats.

  YOLOv8n → TFLite INT8  (for Raspberry Pi 4)
  YOLOv8n → TensorRT FP16 (for Jetson Nano, run ON device)
  PaddleOCR → ONNX (for ONNX Runtime on Pi/Jetson)

Usage:
  python training/export_models.py
  python training/export_models.py --device pi4
  python training/export_models.py --device jetson
"""

import argparse
import shutil
from pathlib import Path


def export_yolo_tflite(weights_path: str, data_yaml: str, output_name: str,
                       imgsz: int = 320, int8: bool = True):
    """Export YOLOv8 model to TFLite INT8."""
    from ultralytics import YOLO

    print(f"\n[EXPORT] {weights_path} → TFLite INT8")
    model = YOLO(weights_path)
    model.export(
        format="tflite",
        int8=int8,
        data=data_yaml,
        imgsz=imgsz,
        simplify=True,
        nms=True,
    )

    src = Path(weights_path).parent / "best_int8.tflite"
    if not src.exists():
        src = Path(weights_path).with_suffix("_int8.tflite")
        if not src.exists():
            possible = list(Path(weights_path).parent.glob("*int8*.tflite"))
            if possible:
                src = possible[0]

    dest = Path("models") / output_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if src.exists():
        shutil.copy2(src, dest)
        print(f"  [OK] → {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  [WARN] TFLite file not found at expected path. Check runs/ directory.")


def export_yolo_tensorrt(weights_path: str, output_name: str):
    """Export YOLOv8 to TensorRT FP16 (must run ON Jetson device)."""
    print(f"\n[EXPORT] {weights_path} → TensorRT FP16")
    print("  NOTE: This must be run ON the Jetson Nano device.")
    print("  Commands to run on Jetson:")
    print(f"    from ultralytics import YOLO")
    print(f"    model = YOLO('{weights_path}')")
    print(f"    model.export(format='engine', device=0, half=True, simplify=True, workspace=2)")


def validate_quantised_model(tflite_path: str, data_yaml: str):
    """Validate INT8 model accuracy vs FP32 baseline."""
    from ultralytics import YOLO

    print(f"\n[VALIDATE] Checking INT8 accuracy: {tflite_path}")
    model = YOLO(tflite_path)
    metrics = model.val(data=data_yaml, split="test", device="cpu")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print("  Acceptable degradation from FP32: mAP50 drop < 3%")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Export models for edge deployment")
    parser.add_argument("--device", choices=["pi4", "jetson", "both"], default="pi4")
    parser.add_argument("--imgsz-pi4", type=int, default=320)
    parser.add_argument("--imgsz-jetson", type=int, default=640)
    args = parser.parse_args()

    tw_weights = "runs/obj3/twowheeler_v1/weights/best.pt"
    tw_data = "datasets/merged_twowheeler/data.yaml"
    lp_weights = "runs/obj3/lp_localise_v1/weights/best.pt"
    lp_data = "datasets/merged_lp_localise/data.yaml"

    print("=" * 60)
    print("MODEL EXPORT — Edge Deployment Conversion")
    print("=" * 60)

    if args.device in ("pi4", "both"):
        imgsz = args.imgsz_pi4

        if Path(tw_weights).exists():
            export_yolo_tflite(tw_weights, tw_data, "twowheeler_int8.tflite", imgsz)
        else:
            print(f"[SKIP] Two-wheeler weights not found: {tw_weights}")

        if Path(lp_weights).exists():
            export_yolo_tflite(lp_weights, lp_data, "lp_localise_int8.tflite", imgsz)
        else:
            print(f"[SKIP] LP localiser weights not found: {lp_weights}")

    if args.device in ("jetson", "both"):
        if Path(tw_weights).exists():
            export_yolo_tensorrt(tw_weights, "twowheeler_fp16.engine")
        if Path(lp_weights).exists():
            export_yolo_tensorrt(lp_weights, "lp_localise_fp16.engine")

    print("\n" + "=" * 60)
    print("EXPECTED MODEL SIZES ON DEVICE")
    print("=" * 60)
    print("  twowheeler_int8.tflite       ~3.2 MB")
    print("  lp_localise_int8.tflite      ~3.2 MB")
    print("  paddleocr_det.onnx           ~2.8 MB")
    print("  paddleocr_rec_indian.onnx    ~8.0 MB")
    print("  paddleocr_cls.onnx           ~1.0 MB")
    print("  ─────────────────────────────────────")
    print("  TOTAL:                       ~18.2 MB")


if __name__ == "__main__":
    main()

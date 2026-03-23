"""
Training Step 3 — Fine-tune PaddleOCR PP-OCRv3 on Indian Licence Plate Data.

Fine-tunes the recognition (character reading) component of PaddleOCR.
The detection component is already strong from base model.

Target metrics:
  Character accuracy > 94%
  Word accuracy (full plate) > 85%

Usage:
  python training/finetune_paddleocr.py
  python training/finetune_paddleocr.py --epochs 60 --batch 128

Prerequisites:
  pip install paddlepaddle-gpu paddleocr
  Download base model weights (done automatically if missing)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_REC_MODEL_URL = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar"
)


def ensure_base_model(model_dir: str = "models/ppocr_base"):
    """Download PP-OCRv3 English recognition base model if not present."""
    model_path = Path(model_dir)
    if model_path.exists() and any(model_path.iterdir()):
        print(f"[OK] Base model found at {model_path}")
        return str(model_path / "best_accuracy")

    print("[DOWNLOAD] Downloading PP-OCRv3 English rec base model...")
    model_path.mkdir(parents=True, exist_ok=True)

    tar_file = model_path / "en_PP-OCRv3_rec_train.tar"
    subprocess.run(["curl", "-L", "-o", str(tar_file), BASE_REC_MODEL_URL], check=True)
    subprocess.run(["tar", "-xf", str(tar_file), "-C", str(model_path)], check=True)

    print(f"[OK] Base model extracted to {model_path}")
    return str(model_path / "en_PP-OCRv3_rec_train" / "best_accuracy")


def create_char_dict(output_path: str = "datasets/ocr_combined/indian_lp_char_dict.txt"):
    """Create character dictionary for Indian LP recognition."""
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for c in chars:
            f.write(c + "\n")
    print(f"[OK] Character dict saved: {output_path} ({len(chars)} chars)")
    return output_path


def create_training_config(
    base_model: str,
    train_labels: str,
    val_labels: str,
    char_dict: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
) -> str:
    """Generate PaddleOCR training YAML config."""
    config = f"""Global:
  use_gpu: true
  epoch_num: {epochs}
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {output_dir}
  save_epoch_step: 5
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: {base_model}
  checkpoints:
  use_visualdl: false
  infer_img: null
  character_dict_path: {char_dict}
  max_text_length: 15
  infer_mode: false
  use_space_char: false
  save_res_path: {output_dir}/rec_results.txt

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 64
            depth: 2
            hidden_dims: 120
            use_guide: true
          Head:
            fc_decay: 0.00001
      - SARHead:
          enc_dim: 512
          max_text_length: 15

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - SARLoss:

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0005
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 0.00001

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
      - {train_labels}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug:
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']
  loader:
    shuffle: true
    batch_size_per_card: {batch_size}
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
      - {val_labels}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: {batch_size}
    num_workers: 2
"""
    config_path = Path(output_dir) / "finetune_config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config)

    print(f"[OK] Training config saved: {config_path}")
    return str(config_path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PaddleOCR on Indian LP data")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--train-labels", default="datasets/ocr_combined/train_labels.txt")
    parser.add_argument("--val-labels", default="datasets/ocr_combined/val_labels.txt")
    parser.add_argument("--output-dir", default="output/paddleocr_indian_rec")
    args = parser.parse_args()

    base_model = ensure_base_model()
    char_dict = create_char_dict()

    config_path = create_training_config(
        base_model=base_model,
        train_labels=args.train_labels,
        val_labels=args.val_labels,
        char_dict=char_dict,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
    )

    print("\n" + "=" * 60)
    print("PADDLEOCR FINE-TUNING")
    print("=" * 60)
    print(f"Config:      {config_path}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch}")
    print(f"Output:      {args.output_dir}")
    print()
    print("To start training, run:")
    print(f"  python tools/train.py -c {config_path}")
    print()
    print("After training, export for inference:")
    print(f"  python tools/export_model.py -c {config_path} \\")
    print(f"    -o Global.pretrained_model={args.output_dir}/best_accuracy \\")
    print(f"       Global.save_inference_dir=models/paddleocr_rec_indian/")
    print()
    print("Then convert to ONNX:")
    print("  paddle2onnx --model_dir models/paddleocr_rec_indian/ \\")
    print("    --model_filename inference.pdmodel \\")
    print("    --params_filename inference.pdiparams \\")
    print("    --save_file models/paddleocr_rec_indian.onnx \\")
    print("    --opset_version 11")


if __name__ == "__main__":
    main()

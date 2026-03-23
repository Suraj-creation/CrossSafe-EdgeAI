# Datasets (Objective 3)

Large **image** folders are listed in the repo root `.gitignore` to keep the GitHub repository under size limits.

## Regenerate locally

1. **Synthetic licence plates** (OCR / LP training aid):  
   `python scripts/generate_synthetic_plates.py`

2. **YOLO training packs** (two-wheeler + LP composite scenes):  
   `python scripts/prepare_training_data.py`

3. **Public corpora** (optional):  
   `python scripts/download_datasets.py --group all`

Label files (`labels/**/*.txt`) and `data.yaml` files are tracked where present so splits and class maps stay reproducible.

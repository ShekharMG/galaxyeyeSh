# SAR-EO Building Damage Change Detection

Binary pixel-level change detection between pre-event RGB electro-optical (EO) imagery and post-event grayscale Synthetic Aperture Radar (SAR) imagery using a Siamese UNet architecture.

---

## Project Description

This project implements an end-to-end deep learning pipeline for detecting building damage by comparing pre-disaster EO images with post-disaster SAR images. The model predicts a binary change mask where each pixel is classified as **No-Change (0)** or **Change (1)**.

### Key Features

- **Siamese UNet** with shared encoder (ResNet34, ImageNet pretrained) and separate input stems for RGB (3-ch) and SAR (1-ch) inputs
- **Feature fusion** via concatenation + absolute difference at each encoder scale
- **Class imbalance handling** with auto-computed positive class weights and Dice + BCE combined loss
- **Selective augmentations**: spatial transforms applied to all images, color jitter applied only to EO images
- **Label remapping**: original 4-class labels mapped to binary (0,1 → No-Change; 2,3 → Change)

---

## Requirements

- Python 3.10+
- GPU (recommended) — CUDA-capable with ~16 GB VRAM for batch_size=16 at 256×256
- **CPU-only is supported** but significantly slower (see [Training Notes](#training-notes))

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## Dataset Structure

```
./
├── train/
│   └── train/
│       ├── pre-event/      # RGB EO images (.tif)
│       ├── post-event/     # Grayscale SAR images (.tif)
│       └── target/         # Annotation masks (.tif)
├── val/
│   └── val/
│       ├── pre-event/
│       ├── post-event/
│       └── target/
└── test/
    └── test/
        ├── pre-event/
        ├── post-event/
        └── target/
```

**Label Remapping:**
| Original Label | Binary Label | Meaning    |
|:--------------:|:------------:|:----------:|
| 0              | 0            | No-Change  |
| 1              | 0            | No-Change  |
| 2              | 1            | Change     |
| 3              | 1            | Change     |

**Dataset Split Sizes:**
| Split | Samples |
|:-----:|:-------:|
| Train | 2,781   |
| Val   | 334     |
| Test  | 77      |

---

## Training Notes

> ⚠️ **CPU-Only Training**: This project was trained entirely on a **CPU with no dedicated GPU**.
> Training 50 epochs on a CPU would take approximately **4 days** on a standard laptop.
> To keep training feasible, we limited training to **3 epochs** with a reduced image size of **128×128** and a lighter backbone (**ResNet18** instead of ResNet34).
> The metrics reflect this early-stopping constraint — a GPU with more epochs would yield significantly better results.

### Recommended full-scale config (GPU required):
| Parameter    | CPU Run (this project) | Recommended (GPU) |
|:-------------|:----------------------:|:-----------------:|
| `encoder`    | resnet18               | resnet34          |
| `image_size` | 128                    | 256               |
| `epochs`     | 3                      | 50                |
| `batch_size` | 16                     | 16                |

```bash
python train.py --config config.yaml
```

Checkpoints are saved to `checkpoints/`:
- `best_model.pth` — best validation F1
- `latest_model.pth` — most recent epoch

---

## Evaluation

```bash
# Evaluate on test set
python eval.py --data_path ./ --weights checkpoints/best_model.pth --split test

# Evaluate on validation set
python eval.py --data_path ./ --weights checkpoints/best_model.pth --split val
```

Results are saved to `eval_results/`:
- `confusion_matrix.png` — confusion matrix visualization
- `vis_001.png` ... `vis_005.png` — qualitative comparison (Pre | Post | GT | Pred)

---

## Model Weights

`checkpoints/best_model.pth` — Trained for **3 epochs** on CPU (ResNet18 backbone, 128×128)

---

## Results

> Results after **3 epochs** of CPU-only training (ResNet18, 128×128 images).
> High Recall (0.94) indicates the model correctly identifies most changed pixels,
> but low Precision (0.06) shows many false positives — expected at this early training stage with no GPU.

| Split | IoU    | Precision | Recall | F1     |
|:-----:|:------:|:---------:|:------:|:------:|
| Val   | 0.0647 | 0.0649    | 0.9441 | 0.1215 |
| Test  | —      | —         | —      | —      |

### Confusion Matrix (Val Split — 334 samples)

|                | Pred = No-Change | Pred = Change |
|:---------------|:----------------:|:-------------:|
| **GT = No-Change** | 3,713,522    | 1,638,249     |
| **GT = Change**    | 6,730        | 113,755       |

> 💡 **Interpretation**: The model has very high recall (catches 94% of actual changes) but generates many false positives. More training epochs and a GPU would dramatically improve precision and IoU.

---

## Project Files

| File               | Description                                           |
|:-------------------|:------------------------------------------------------|
| `dataset.py`       | PyTorch Dataset with label remapping & augmentations   |
| `model.py`         | Siamese UNet with shared encoder & feature fusion      |
| `loss.py`          | Combined Dice + weighted BCE loss                      |
| `train.py`         | Training script with AdamW + cosine LR scheduler       |
| `eval.py`          | Evaluation with metrics, confusion matrix, and visuals |
| `config.yaml`      | All configurable hyperparameters                       |
| `requirements.txt` | Pinned Python dependencies                             |

---

## References

1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
2. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
3. Daudt et al., "Fully Convolutional Siamese Networks for Change Detection", ICIP 2018
4. [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
5. [Albumentations](https://albumentations.ai/)

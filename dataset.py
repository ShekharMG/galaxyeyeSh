"""
dataset.py — PyTorch Dataset for SAR-EO Change Detection

Loads triplets of (pre-event EO, post-event SAR, target mask) and applies:
  - Label remapping: {0,1} → 0 (No-Change), {2,3} → 1 (Change)
  - Albumentations augmentations (spatial applied to all, color applied to EO only)
  - Resizing to configured image_size

Key design decisions:
  - Uses `additional_targets` in Albumentations to keep spatial transforms in sync
    across pre, post, and mask images.
  - Brightness/contrast augmentation is applied ONLY to the EO (pre-event) image
    because SAR radiometry is physically different and shouldn't be color-jittered.
  - Images are loaded with PIL (supports .tif) and converted to numpy for Albumentations.
  - The actual data directory is nested (e.g. train/train/pre-event/), so we handle
    both flat and nested layouts automatically.
"""

import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Label remapping lookup: original label → binary label
#   0 → 0  (No-Change / background)
#   1 → 0  (No-Change / minor damage treated as no change)
#   2 → 1  (Change / significant damage)
#   3 → 1  (Change / destroyed)
# Any label ≥ 4 is clipped to 1 as a safety measure.
# ---------------------------------------------------------------------------
LABEL_REMAP = np.array([0, 0, 1, 1], dtype=np.uint8)


def remap_labels(mask: np.ndarray) -> np.ndarray:
    """Remap multi-class labels to binary (0=No-Change, 1=Change)."""
    # Clip to valid range then apply lookup table
    clipped = np.clip(mask, 0, len(LABEL_REMAP) - 1)
    return LABEL_REMAP[clipped]


def _resolve_split_dir(data_root: str, split: str) -> str:
    """
    Resolve the actual directory for a split.
    Handles both flat layout  (data_root/train/pre-event/)
    and nested layout          (data_root/train/train/pre-event/)
    by checking which one actually contains the expected subdirectories.
    """
    flat = os.path.join(data_root, split)
    nested = os.path.join(data_root, split, split)

    # Prefer the path that contains pre-event/post-event/target
    if os.path.isdir(os.path.join(nested, "pre-event")):
        return nested
    elif os.path.isdir(os.path.join(flat, "pre-event")):
        return flat
    else:
        raise FileNotFoundError(
            f"Cannot find pre-event/ directory in either {flat} or {nested}. "
            f"Please check your data_root ('{data_root}') and split ('{split}')."
        )


def get_augmentations(cfg: dict, is_train: bool = True):
    """
    Build an Albumentations Compose pipeline.

    - Spatial transforms (flip, rotate) are applied to all images + mask together.
    - Brightness/contrast is applied ONLY to the pre-event EO image via a separate
      transform that we apply manually in __getitem__.
    - Resize is always applied.

    Args:
        cfg: parsed config dict (from config.yaml)
        is_train: whether to include stochastic augmentations
    """
    img_size = cfg.get("image_size", 256)
    aug_cfg = cfg.get("augmentations", {})

    spatial_transforms = [
        A.Resize(img_size, img_size),  # Always resize to target
    ]

    if is_train:
        if aug_cfg.get("horizontal_flip", False):
            spatial_transforms.append(A.HorizontalFlip(p=0.5))
        if aug_cfg.get("vertical_flip", False):
            spatial_transforms.append(A.VerticalFlip(p=0.5))
        if aug_cfg.get("rotation", 0) > 0:
            limit = aug_cfg["rotation"]
            spatial_transforms.append(A.RandomRotate90(p=0.5))

    # Normalize + ToTensor at the end
    spatial_transforms.extend([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],  # We normalize per-channel in __getitem__
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # `additional_targets` keeps spatial transforms in sync for post-event image
    transform = A.Compose(
        spatial_transforms,
        additional_targets={"post": "image"},
    )

    # Separate color jitter for EO only (applied before spatial)
    eo_color_transform = None
    if is_train and aug_cfg.get("brightness_contrast", False):
        eo_color_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
        ])

    return transform, eo_color_transform


class ChangeDetectionDataset(Dataset):
    """
    Dataset for binary change detection from pre-event EO + post-event SAR.

    Returns a dict:
        {
            "pre":  FloatTensor  [3, H, W]  — pre-event EO (RGB)
            "post": FloatTensor  [1, H, W]  — post-event SAR (grayscale)
            "mask": LongTensor   [H, W]     — binary mask (0=No-Change, 1=Change)
        }
    """

    def __init__(self, data_root: str, split: str, cfg: dict, is_train: bool = True):
        """
        Args:
            data_root: root directory containing train/val/test folders
            split: one of 'train', 'val', 'test'
            cfg: parsed config dict
            is_train: enable stochastic augmentations
        """
        super().__init__()
        self.split = split
        self.cfg = cfg
        self.img_size = cfg.get("image_size", 256)

        # Resolve potentially nested directory structure
        split_dir = _resolve_split_dir(data_root, split)

        self.pre_dir = os.path.join(split_dir, "pre-event")
        self.post_dir = os.path.join(split_dir, "post-event")
        self.target_dir = os.path.join(split_dir, "target")

        # Collect filenames — support both .tif and .png
        self.filenames = sorted([
            f for f in os.listdir(self.pre_dir)
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
        ])

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No images found in {self.pre_dir}. "
                "Supported formats: .tif, .tiff, .png, .jpg, .jpeg"
            )

        print(f"[Dataset] {split}: {len(self.filenames)} samples from {split_dir}")

        # Build augmentation pipelines
        self.transform, self.eo_color_transform = get_augmentations(cfg, is_train)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # --- Load images and resize efficiently BEFORE numpy conversion ---
        # Pre-event: RGB EO → 3 channels
        pre_img = Image.open(os.path.join(self.pre_dir, fname)).convert("RGB")
        pre_img = pre_img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        pre_img = np.array(pre_img, dtype=np.uint8)  # (H, W, 3)

        # Post-event: Grayscale SAR → 1 channel
        post_img = Image.open(os.path.join(self.post_dir, fname)).convert("L")
        post_img = post_img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        post_img = np.array(post_img, dtype=np.uint8)  # (H, W)
        # Expand to (H, W, 3) for Albumentations compatibility (will be reduced later)
        post_img_3ch = np.stack([post_img] * 3, axis=-1)

        # Target mask
        mask = Image.open(os.path.join(self.target_dir, fname)).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        mask = np.array(mask, dtype=np.uint8)  # (H, W)

        # --- Apply label remapping BEFORE any augmentation ---
        mask = remap_labels(mask)

        # --- Apply EO-only color augmentation (before spatial) ---
        if self.eo_color_transform is not None:
            augmented_eo = self.eo_color_transform(image=pre_img)
            pre_img = augmented_eo["image"]

        # --- Apply synchronized spatial augmentation ---
        augmented = self.transform(
            image=pre_img,
            post=post_img_3ch,
            mask=mask,
        )

        pre_tensor = augmented["image"]           # (3, H, W) float
        post_tensor = augmented["post"]           # (3, H, W) float — take only ch0
        mask_tensor = augmented["mask"]           # (H, W) uint8

        # Reduce post back to single channel
        post_tensor = post_tensor[0:1, :, :]      # (1, H, W)

        return {
            "pre": pre_tensor.float(),
            "post": post_tensor.float(),
            "mask": mask_tensor.long(),
        }


def compute_pos_weight(dataset: ChangeDetectionDataset, max_samples: int = 500) -> float:
    """
    Estimate the positive-class weight for BCE loss by scanning a subset of
    the training set.  This counteracts the severe class imbalance where
    'change' pixels (label=1) are much rarer than 'no-change' pixels (label=0).

    pos_weight = num_negative / num_positive

    Args:
        dataset: a ChangeDetectionDataset instance (training split)
        max_samples: cap on how many images to scan (for speed)

    Returns:
        pos_weight (float): weight for the positive (change) class
    """
    num_pos = 0
    num_neg = 0
    n = min(len(dataset), max_samples)

    print(f"[pos_weight] Scanning {n} samples to estimate class balance...")

    for i in range(n):
        fname = dataset.filenames[i]
        mask = Image.open(os.path.join(dataset.target_dir, fname)).convert("L")
        mask = np.array(mask, dtype=np.uint8)
        mask = remap_labels(mask)

        num_pos += np.sum(mask == 1)
        num_neg += np.sum(mask == 0)

    if num_pos == 0:
        print("[pos_weight] WARNING: No positive pixels found! Using pos_weight=1.0")
        return 1.0

    weight = num_neg / num_pos
    print(f"[pos_weight] neg={num_neg:,}  pos={num_pos:,}  ratio={weight:.2f}")
    return float(weight)

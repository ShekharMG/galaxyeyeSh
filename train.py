"""
train.py — Training script for SAR-EO Change Detection

Usage:
    python train.py --config config.yaml

Features:
  - Loads config from YAML file
  - Computes pos_weight automatically from training set if config says 'auto'
  - AdamW optimizer with cosine annealing LR scheduler
  - Validates every epoch, saves best checkpoint by F1 score
  - Logs loss, IoU, Precision, Recall, F1 per epoch
"""

import os
import argparse
import random
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import ChangeDetectionDataset, compute_pos_weight
from model import build_model
from loss import build_loss


def set_seed(seed: int):
    """Reproducibility: fix all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision
        if scaler is not None and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(pre, post)
                loss = criterion(logits, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pre, post)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * pre.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and return average loss + metrics."""
    model.eval()
    total_loss = 0.0
    
    tp = 0.0
    fp = 0.0
    fn = 0.0

    for batch in tqdm(loader, desc="  Val", leave=False):
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(pre, post)
                loss = criterion(logits, mask)
        else:
            logits = model(pre, post)
            loss = criterion(logits, mask)
            
        total_loss += loss.item() * pre.size(0)

        # Binarize predictions at threshold 0.5
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).long()

        # Accumulate metrics per batch to prevent massive memory usage
        tp += ((preds == 1) & (mask == 1)).sum().item()
        fp += ((preds == 1) & (mask == 0)).sum().item()
        fn += ((preds == 0) & (mask == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    metrics = {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train SAR-EO Change Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # --- Create output directory ---
    os.makedirs("checkpoints", exist_ok=True)

    # --- Datasets & Dataloaders ---
    train_ds = ChangeDetectionDataset(cfg["data_root"], "train", cfg, is_train=True)
    val_ds = ChangeDetectionDataset(cfg["data_root"], "val", cfg, is_train=False)

    # Enable faster CUDNN benchmarking for stable input sizes
    torch.backends.cudnn.benchmark = True
    
    # Let dataloader use multiple workers but be safe on Windows by default using config
    num_workers = cfg.get("num_workers", 4 if os.name != 'nt' else 0)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    # --- Compute positive class weight ---
    pos_weight_cfg = cfg.get("pos_weight", "auto")
    if pos_weight_cfg == "auto":
        pos_weight = compute_pos_weight(train_ds, max_samples=500)
    else:
        pos_weight = float(pos_weight_cfg)

    # --- Model, Loss, Optimizer, Scheduler ---
    model = build_model(cfg).to(device)
    criterion = build_loss(pos_weight).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    # --- Training loop ---
    best_f1 = 0.0
    epochs = cfg["epochs"]
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    print(f"\n{'='*60}")
    print(f"  Starting training for {epochs} epochs")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        print(f"Epoch [{epoch}/{epochs}]  lr={optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Log metrics
        print(
            f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"IoU={val_metrics['iou']:.4f}  P={val_metrics['precision']:.4f}  "
            f"R={val_metrics['recall']:.4f}  F1={val_metrics['f1']:.4f}"
        )

        # Save best checkpoint by F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            ckpt_path = os.path.join("checkpoints", "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": cfg,
            }, ckpt_path)
            print(f"  ✓ Saved best model (F1={best_f1:.4f}) → {ckpt_path}")

        # Save latest checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
            "config": cfg,
        }, os.path.join("checkpoints", "latest_model.pth"))

    print(f"\n{'='*60}")
    print(f"  Training complete. Best F1: {best_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

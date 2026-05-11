"""
eval.py — Evaluation script for SAR-EO Change Detection

Usage:
    python eval.py --data_path data/ --weights checkpoints/best_model.pth

Features:
  - Reports IoU, Precision, Recall, F1 for the change class (label=1)
  - Prints confusion matrix
  - Saves 5+ qualitative prediction visualizations
"""

import os
import argparse
import yaml
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import ChangeDetectionDataset
from model import build_model


def compute_metrics(preds, targets):
    """Compute binary metrics for class=1."""
    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Run evaluation and collect predictions."""
    model.eval()
    all_preds = []
    all_targets = []
    # Store per-sample data for visualization
    vis_samples = []

    for batch in tqdm(loader, desc="Evaluating"):
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        mask = batch["mask"]

        logits = model(pre, post)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5).long().cpu()

        all_preds.append(preds.reshape(-1))
        all_targets.append(mask.reshape(-1))

        # Collect samples for visualization (up to 10)
        if len(vis_samples) < 10:
            for i in range(min(pre.size(0), 10 - len(vis_samples))):
                vis_samples.append({
                    "pre": pre[i].cpu(),
                    "post": post[i].cpu(),
                    "gt": mask[i].cpu().numpy(),
                    "pred": preds[i].cpu().numpy(),
                })

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return all_preds, all_targets, vis_samples


def save_visualizations(vis_samples, output_dir, num_vis=5):
    """Save side-by-side comparison images: pre | post | GT | prediction."""
    os.makedirs(output_dir, exist_ok=True)

    for i, sample in enumerate(vis_samples[:num_vis]):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Pre-event (RGB) — denormalize from [0,1] to [0,255]
        pre_img = sample["pre"].permute(1, 2, 0).numpy()
        pre_img = np.clip(pre_img * 255, 0, 255).astype(np.uint8)
        axes[0].imshow(pre_img)
        axes[0].set_title("Pre-event (EO)", fontsize=12)
        axes[0].axis("off")

        # Post-event (SAR grayscale)
        post_img = sample["post"].squeeze(0).numpy()
        post_img = np.clip(post_img * 255, 0, 255).astype(np.uint8)
        axes[1].imshow(post_img, cmap="gray")
        axes[1].set_title("Post-event (SAR)", fontsize=12)
        axes[1].axis("off")

        # Ground truth mask
        axes[2].imshow(sample["gt"], cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[2].set_title("Ground Truth", fontsize=12)
        axes[2].axis("off")

        # Predicted mask
        axes[3].imshow(sample["pred"], cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[3].set_title("Prediction", fontsize=12)
        axes[3].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"vis_{i+1:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved visualization → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Change Detection Model")
    parser.add_argument("--data_path", type=str, required=True, help="Root data directory")
    parser.add_argument("--weights", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    # --- Load checkpoint and config ---
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    print(f"[Eval] Loaded checkpoint from epoch {checkpoint['epoch']} (F1={checkpoint['best_f1']:.4f})")

    # --- Build model and load weights ---
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # --- Dataset ---
    dataset = ChangeDetectionDataset(args.data_path, args.split, cfg, is_train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Evaluate ---
    all_preds, all_targets, vis_samples = evaluate(model, loader, device)

    # --- Metrics ---
    metrics = compute_metrics(all_preds, all_targets)
    print(f"\n{'='*50}")
    print(f"  Results on {args.split} split ({len(dataset)} samples)")
    print(f"{'='*50}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"{'='*50}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_targets.numpy(), all_preds.numpy(), labels=[0, 1])
    print(f"\nConfusion Matrix:")
    print(f"  {'':>12} Pred=0     Pred=1")
    print(f"  {'GT=0':>12} {cm[0,0]:>10,} {cm[0,1]:>10,}")
    print(f"  {'GT=1':>12} {cm[1,0]:>10,} {cm[1,1]:>10,}")

    # Save confusion matrix plot
    os.makedirs(args.output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No-Change", "Change"])
    disp.plot(ax=ax, cmap="Blues", values_format=",")
    plt.title(f"Confusion Matrix ({args.split})", fontsize=14)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved confusion matrix → {cm_path}")

    # --- Visualizations ---
    print(f"\nSaving qualitative visualizations...")
    save_visualizations(vis_samples, args.output_dir, num_vis=5)

    print(f"\n✓ Evaluation complete. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

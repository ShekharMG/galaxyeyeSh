"""
loss.py — Combined Loss for Change Detection

Combines Dice Loss and weighted Binary Cross-Entropy (BCE) to handle
severe class imbalance in change detection tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1).float()
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + weighted BCE loss."""

    def __init__(self, pos_weight: float = 1.0, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target_float = target.float()

        # BCE with logits (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, target_float, pos_weight=self.pos_weight)

        # Dice loss (needs probabilities)
        probs = torch.sigmoid(logits)
        dice = self.dice_loss(probs, target_float)

        return self.bce_weight * bce + self.dice_weight * dice


def build_loss(pos_weight: float = 1.0) -> DiceBCELoss:
    """Factory function to create the loss."""
    print(f"[Loss] DiceBCE with pos_weight={pos_weight:.2f}")
    return DiceBCELoss(pos_weight=pos_weight)

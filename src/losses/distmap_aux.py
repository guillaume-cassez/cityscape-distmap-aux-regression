"""DistMap auxiliary loss (Paper 1).

Masked MSE between the predicted normalized per-class Signed Distance Transform
(an auxiliary head with a tanh output, values in [-1, 1]) and the precomputed SDT
target normalized by `scale`. The int8 SDT cache is clipped to [-127, 127], so
scale=127 maps it to [-1, 1] to match the tanh range. Void pixels (ignore_index in
the label) are excluded.

Mirrors the BRATS *Distance Map Auxiliary Loss* method (MSE on a tanh SDT-regression
head), transposed to 2D / 19 Cityscapes classes. "sans DWA" = fixed weight, applied
in scripts/train.py as `loss + lambda * DistMapAuxLoss(...)`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMapAuxLoss(nn.Module):
    def __init__(self, scale: float = 127.0, ignore_index: int = 255):
        super().__init__()
        self.scale = float(scale)
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, sdt: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """pred: (B,C,H,W) in [-1,1] (tanh). sdt: (B,C,H,W) raw SDT. target: (B,H,W) labels for the void mask."""
        tgt = (sdt.float() / self.scale).clamp(-1.0, 1.0)
        if target is None:
            return F.mse_loss(pred, tgt)
        valid = (target != self.ignore_index).unsqueeze(1).to(pred.dtype)  # (B,1,H,W)
        sq = (pred - tgt) ** 2 * valid
        denom = valid.sum() * pred.shape[1] + 1e-6
        return sq.sum() / denom

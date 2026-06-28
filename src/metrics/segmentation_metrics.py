"""
Segmentation metrics: mIoU, per-class IoU, boundary F1, trimap IoU.

mIoU follows the official Cityscapes definition: a single confusion matrix is
accumulated over the whole evaluation set (dataset-level), per-class
IoU = TP / (TP + FP + FN), averaged over classes that appear in the set; the
void label (`ignore_index`) is excluded.

Boundary F1 and Trimap IoU are computed PER CLASS on binary class masks.
An earlier implementation passed the multi-class label map directly to
`scipy.ndimage.binary_dilation`, which treats any non-zero label as foreground —
so only the road(0)-vs-rest contour was measured, not the inter-class
boundaries. Both metrics now operate per class on binary masks, matching the
"averaged across classes" definition reported in the paper.
"""

import numpy as np
import torch
from typing import Dict
from scipy.ndimage import binary_dilation, binary_erosion

_CONN = np.ones((3, 3), dtype=bool)  # 8-connectivity for morphological ops


class SegmentationMetrics:
    """Accumulate a dataset-level confusion matrix; compute mIoU + per-class IoU."""

    def __init__(self, num_classes: int = 19, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Accumulate a batch. `pred` is logits (N,C,H,W) or labels; `target` is (N,H,W)."""
        pred_np = pred.argmax(dim=1).cpu().numpy() if pred.dim() == 4 else pred.cpu().numpy()
        target_np = target.cpu().numpy()

        # Valid pixels: drop void and any out-of-range label.
        k = (target_np != self.ignore_index) & (target_np >= 0) & (target_np < self.num_classes)
        # Vectorised confusion-matrix update (replaces the per-pixel Python loop:
        # ~3 orders of magnitude faster on 500 full-res val images).
        idx = self.num_classes * target_np[k].astype(np.int64) + pred_np[k].astype(np.int64)
        self.confusion_matrix += np.bincount(
            idx, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        cm = self.confusion_matrix
        intersection = np.diag(cm).astype(np.float64)
        union = (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)).astype(np.float64)
        valid = union > 0  # classes absent from the whole set are excluded (official convention)
        # Exact IoU = TP/(TP+FP+FN) per class, bit-for-bit identical to the official
        # cityscapesScripts definition (validated in tests/test_official_miou.py against
        # cityscapesscripts.evaluation.getIouScoreForLabel). No epsilon smoothing: the
        # `valid` mask already excludes union==0, so the previous `+1e-6` in the
        # denominator was redundant and biased the value away from the official metric.
        iou = np.divide(intersection, union, out=np.zeros_like(union), where=valid)
        miou = iou[valid].mean() if valid.any() else 0.0
        return {
            "mIoU": float(miou),
            "per_class_iou": {i: float(iou[i]) for i in range(self.num_classes)},
        }


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Morphological gradient of a BINARY mask (~2px band).

    Safe because the input is boolean: unlike the previous implementation, this
    is never called on a multi-class label map.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return binary_dilation(mask, _CONN) ^ binary_erosion(mask, _CONN)


def _interclass_boundaries(label: np.ndarray, ignore_index: int) -> np.ndarray:
    """Pixels on a boundary between two DIFFERENT valid classes (4-neighbour test).

    Computed directly on the label map without binarising, so every inter-class
    transition counts — not just class-0-vs-rest. Transitions to/from the void
    label are ignored.
    """
    valid = label != ignore_index
    b = np.zeros(label.shape, dtype=bool)
    dh = (label[:, :-1] != label[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    b[:, :-1] |= dh
    b[:, 1:] |= dh
    dv = (label[:-1, :] != label[1:, :]) & valid[:-1, :] & valid[1:, :]
    b[:-1, :] |= dv
    b[1:, :] |= dv
    return b


def compute_boundary_f1(
    pred: np.ndarray,
    target: np.ndarray,
    distance: int = 3,
    num_classes: int = 19,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """Per-class boundary F1 within a pixel tolerance, averaged over GT classes.

    For each class c, the contours of the binary masks (pred==c) and (target==c)
    are extracted, then precision/recall are matched within `distance` pixels
    (3px, matching the paper). The mean is taken over classes present in the
    ground truth (standard convention); a class predicted but absent from GT is
    penalised by mIoU, not here.
    """
    valid = target != ignore_index
    tol = np.ones((2 * distance + 1, 2 * distance + 1), dtype=bool)

    f1s, precs, recs = [], [], []
    for c in range(num_classes):
        gt_c = (target == c) & valid
        if not gt_c.any():
            continue
        pr_c = (pred == c) & valid

        gt_b = _binary_boundary(gt_c)
        pr_b = _binary_boundary(pr_c)
        if gt_b.sum() == 0 and pr_b.sum() == 0:
            f1s.append(1.0); precs.append(1.0); recs.append(1.0)
            continue

        gt_b_dil = binary_dilation(gt_b, tol)
        pr_b_dil = binary_dilation(pr_b, tol)
        precision = (pr_b & gt_b_dil).sum() / (pr_b.sum() + 1e-6)
        recall = (gt_b & pr_b_dil).sum() / (gt_b.sum() + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        f1s.append(f1); precs.append(precision); recs.append(recall)

    return {
        "boundary_f1": float(np.mean(f1s)) if f1s else 0.0,
        "boundary_precision": float(np.mean(precs)) if precs else 0.0,
        "boundary_recall": float(np.mean(recs)) if recs else 0.0,
    }


def compute_trimap_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 19,
    width: int = 3,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """mIoU restricted to a band around ALL inter-class boundaries.

    The trimap band is built from inter-class label boundaries (every class
    transition, not just road-vs-rest), dilated by `width` pixels. Per-class IoU
    is then computed inside that band and averaged over present classes.
    """
    valid = target != ignore_index
    boundary = _interclass_boundaries(target, ignore_index)
    band = np.ones((2 * width + 1, 2 * width + 1), dtype=bool)
    trimap = binary_dilation(boundary, band) & valid
    if trimap.sum() == 0:
        return {"trimap_mIoU": 0.0}

    ious = []
    for c in range(num_classes):
        pred_c = (pred == c) & trimap
        target_c = (target == c) & trimap
        inter = (pred_c & target_c).sum()
        uni = (pred_c | target_c).sum()
        if uni > 0:
            ious.append(inter / uni)
    return {"trimap_mIoU": float(np.mean(ious)) if ious else 0.0}

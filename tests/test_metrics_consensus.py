#!/usr/bin/env python3
"""
Synthetic unit tests for the metric fixes and the consensus filter.

Run from the project root:
    python tests/test_metrics_consensus.py

No GPU and no checkpoints needed — these prove the *code* is correct on crafted
inputs. The corrected numbers on the real val set still require re-running
evaluate.py on the Tower (T7 + GPU).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.segmentation_metrics import (
    compute_boundary_f1,
    compute_trimap_iou,
    _interclass_boundaries,
)
from src.postprocessing.consensus import (
    cc_veto,
    majority_vote,
    agreement_map,
    count_fragments,
)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond):
    results.append((name, cond))
    print(f"  [{PASS if cond else FAIL}] {name}")
    return cond


# ---------------------------------------------------------------------------
# 1. Metric bug: boundary between two NON-road classes must be detected.
# ---------------------------------------------------------------------------
print("\n[1] Boundary metric — inter-class contour (building vs vegetation, no road)")
H = W = 16
target = np.full((H, W), 2, dtype=np.int64)   # building = trainId 2
target[:, W // 2:] = 8                          # vegetation = trainId 8 (right half)

# The OLD implementation ran morphology on the multi-class label map directly.
old_struct = np.ones((3, 3), dtype=bool)
old_boundary = binary_dilation(target, old_struct) != binary_erosion(target, old_struct)
# At the interior building/vegetation seam (cols 7-8, away from image border) the
# old method sees nothing: both labels are non-zero -> all True -> zero gradient.
check("OLD method misses interior inter-class seam",
      not old_boundary[2:H - 2, W // 2 - 1:W // 2 + 1].any())

# The NEW helper detects the seam.
new_boundary = _interclass_boundaries(target, ignore_index=255)
check("NEW _interclass_boundaries detects the seam",
      new_boundary[2:H - 2, W // 2 - 1:W // 2 + 1].any())

# Boundary F1: perfect prediction -> 1.0; a 1px-shifted seam -> < 1.0.
perfect = compute_boundary_f1(target.copy(), target, num_classes=19)
check("boundary_f1 == 1.0 for perfect prediction", abs(perfect["boundary_f1"] - 1.0) < 1e-6)

shifted = target.copy()
shifted[:, W // 2] = 2  # push the building one column into vegetation
err = compute_boundary_f1(shifted, target, num_classes=19)
check("boundary_f1 < 1.0 when the inter-class seam is wrong (OLD gave 1.0)",
      err["boundary_f1"] < 1.0)


# ---------------------------------------------------------------------------
# 2. Trimap IoU built on all inter-class boundaries, not road-only.
# ---------------------------------------------------------------------------
print("\n[2] Trimap IoU — band covers the inter-class seam")
tri_perfect = compute_trimap_iou(target.copy(), target, num_classes=19, width=3)
check("trimap_mIoU == 1.0 for perfect prediction", abs(tri_perfect["trimap_mIoU"] - 1.0) < 1e-6)
tri_err = compute_trimap_iou(shifted, target, num_classes=19, width=3)
check("trimap_mIoU < 1.0 with a seam error", tri_err["trimap_mIoU"] < 1.0)


# ---------------------------------------------------------------------------
# 3. cc_veto: drop an uncorroborated fragment, reassign to the veto label.
# ---------------------------------------------------------------------------
print("\n[3] cc_veto — fragment removal + reassignment")
primary = np.full((20, 20), 13, dtype=np.int64)  # car everywhere
veto = np.full((20, 20), 13, dtype=np.int64)     # veto agrees on car
primary[2:5, 2:5] = 14                            # a 3x3 'truck' blob, NOT in veto
filt = cc_veto(primary, veto, num_classes=19, protect_classes=())
check("uncorroborated truck fragment removed", (filt == 14).sum() == 0)
check("removed pixels reassigned to veto label (car)", (filt[2:5, 2:5] == 13).all())
check("input not mutated", (primary[2:5, 2:5] == 14).all())

# Corroborated fragment is kept.
veto2 = veto.copy()
veto2[3, 3] = 14  # one overlapping truck pixel in the veto
keep = cc_veto(primary, veto2, num_classes=19, protect_classes=())
check("corroborated fragment kept (>=1 overlap pixel)", (keep == 14).sum() == 9)


# ---------------------------------------------------------------------------
# 4. Thin-class protection + max_drop_size guard.
# ---------------------------------------------------------------------------
print("\n[4] Thin-class protection and size guard")
prim_pole = np.full((20, 20), 8, dtype=np.int64)  # vegetation background
prim_pole[:, 10] = 5                               # a thin vertical pole (trainId 5)
veto_none = np.full((20, 20), 8, dtype=np.int64)   # veto has no pole
prot = cc_veto(prim_pole, veto_none, num_classes=19)             # default protects 5
check("pole protected by default (thin class)", (prot == 5).sum() == 20)
unprot = cc_veto(prim_pole, veto_none, num_classes=19, protect_classes=())
check("pole removed when protection disabled", (unprot == 5).sum() == 0)

# Large uncorroborated component survives max_drop_size; small one does not.
prim_big = np.full((20, 20), 13, dtype=np.int64)
prim_big[0:8, 0:8] = 16   # 64px 'train' blob, not in veto
veto_b = np.full((20, 20), 13, dtype=np.int64)
small_dropped = cc_veto(prim_big, veto_b, num_classes=19, protect_classes=(), max_drop_size=10)
check("large blob kept under max_drop_size=10", (small_dropped == 16).sum() == 64)
all_dropped = cc_veto(prim_big, veto_b, num_classes=19, protect_classes=(), max_drop_size=None)
check("large blob dropped when max_drop_size=None (BRATS-strict)", (all_dropped == 16).sum() == 0)


# ---------------------------------------------------------------------------
# 5. Majority vote + agreement map.
# ---------------------------------------------------------------------------
print("\n[5] Majority vote and agreement map (n=3 ensemble)")
p1 = np.full((4, 4), 13, dtype=np.int64)
p2 = np.full((4, 4), 13, dtype=np.int64)
p3 = np.full((4, 4), 14, dtype=np.int64)
p3[0, 0] = 13
voted = majority_vote([p1, p2, p3], num_classes=19)
check("majority vote -> car everywhere (2 of 3)", (voted == 13).all())
agree = agreement_map([p1, p2, p3], num_classes=19)
check("agreement = 1.0 where unanimous", abs(agree[0, 0] - 1.0) < 1e-6)
check("agreement = 2/3 where one voter dissents", abs(agree[1, 1] - 2 / 3) < 1e-6)


# ---------------------------------------------------------------------------
# 6. Fragment count (BRATS-style spatial coherence proxy).
# ---------------------------------------------------------------------------
print("\n[6] Fragment count")
frag = np.full((20, 20), 8, dtype=np.int64)
frag[1:4, 1:4] = 13   # car blob 1
frag[15:18, 15:18] = 13  # car blob 2 (disconnected)
counts = count_fragments(frag, num_classes=19)
check("two disconnected car blobs -> 2 components", counts.get(13) == 2)
check("consensus lowers fragment count",
      count_fragments(cc_veto(frag, np.full((20, 20), 8, dtype=np.int64),
                              num_classes=19, protect_classes=()),
                      num_classes=19).get(13, 0) == 0)


# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
n_pass = sum(c for _, c in results)
n_total = len(results)
print(f"RESULT: {n_pass}/{n_total} checks passed")
sys.exit(0 if n_pass == n_total else 1)

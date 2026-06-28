#!/usr/bin/env python3
"""
Official-metric validation: prove the project's mIoU
(`src.metrics.SegmentationMetrics`) is numerically identical to the OFFICIAL
Cityscapes evaluation routine that the benchmark server runs —
`cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling.getIouScoreForLabel`
+ `getScoreAverage`.

Why this exists (project rule): a benchmark metric must be validated against the
official reference implementation, never trusted as a hand-rolled reimplementation.
mIoU is the *only* official Cityscapes ranking metric for pixel-level semantic
labeling (IoU = TP/(TP+FP+FN) accumulated over the whole set, averaged over the 19
trainId evaluation classes; every pixel whose GT label is `ignoreInEval` — our
trainId 255 — is excluded). Cordts et al., CVPR 2016; cityscapes-dataset.com/benchmarks.

The project metric works in trainId space (19x19 confusion matrix, 255 ignored); the
official routine works in labelId space (the 19 non-ignoreInEval labelIds map
bijectively to trainIds 0..18, the ignoreInEval ids map to 255). This test maps a
synthetic scene into both spaces and asserts the two mIoU values are bit-identical.

Run from the project root:
    python3 tests/test_official_miou.py
Needs: pip install cityscapesscripts   (numpy + torch already required by the project)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import SegmentationMetrics

import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cseval
from cityscapesscripts.helpers.labels import trainId2label, id2label

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, extra=""):
    results.append(bool(cond))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({extra})" if extra else ""))
    return cond


# trainId 0..18 -> a representative (non-ignoreInEval) labelId; 255 -> an ignoreInEval id.
TRAINID2ID = {t: trainId2label[t].id for t in range(19)}
IGNORE_ID = 0  # 'unlabeled', ignoreInEval=True
assert id2label[IGNORE_ID].ignoreInEval, "labelId 0 must be ignoreInEval"


def project_miou(pred_tid: np.ndarray, gt_tid: np.ndarray) -> float:
    m = SegmentationMetrics(num_classes=19, ignore_index=255)
    m.update(torch.from_numpy(pred_tid)[None], torch.from_numpy(gt_tid)[None])
    return m.compute()["mIoU"]


def official_miou(pred_tid: np.ndarray, gt_tid: np.ndarray) -> float:
    """mIoU via the official cityscapesScripts routine, in labelId space."""
    args = argparse.Namespace()
    conf = cseval.generateMatrix(args)  # zeros((maxId+1)^2); also fills args.evalLabels
    g = np.vectorize(lambda v: TRAINID2ID.get(int(v), IGNORE_ID))(gt_tid).astype(np.int64)
    # Real models argmax over 19 logits (never 255); a 255 here only occurs on ignored
    # GT pixels (excluded anyway) -> map any non-eval value to a valid eval id.
    p = np.vectorize(lambda v: TRAINID2ID.get(int(v), TRAINID2ID[0]))(pred_tid).astype(np.int64)
    n = conf.shape[0]
    conf += np.bincount(g.ravel() * n + p.ravel(), minlength=n * n).reshape(n, n).astype(conf.dtype)
    iou = {lab: cseval.getIouScoreForLabel(lab, conf, args) for lab in args.evalLabels}
    return cseval.getScoreAverage(iou, args)


TOL = 1e-9  # both compute float64 tp/union on identical integers -> bit-identical
rng = np.random.default_rng(0)
H = W = 96

# ---------------------------------------------------------------------------
print("\n[1] random scene, all 19 classes present, 255 ignore band, 30% wrong pixels")
gt = rng.integers(0, 19, size=(H, W)).astype(np.int64)
for t in range(19):
    gt.flat[t] = t                       # guarantee every class appears
gt[:, :6] = 255                          # ignore band
pred = gt.copy()
flip = rng.random((H, W)) < 0.30
pred[flip] = rng.integers(0, 19, size=int(flip.sum()))
pred[gt == 255] = rng.integers(0, 19, size=int((gt == 255).sum()))  # arbitrary on ignored GT
pm, om = project_miou(pred, gt), official_miou(pred, gt)
check("project mIoU == official mIoU", abs(pm - om) < TOL, f"proj={pm:.12f} off={om:.12f} d={abs(pm-om):.2e}")

# ---------------------------------------------------------------------------
print("\n[2] perfect prediction -> mIoU == 1.0 in both")
pm, om = project_miou(gt.copy(), gt), official_miou(gt.copy(), gt)
check("both equal 1.0 on valid pixels", abs(pm - 1.0) < TOL and abs(om - 1.0) < TOL, f"proj={pm} off={om}")

# ---------------------------------------------------------------------------
print("\n[3] a class absent from GT but predicted -> IoU=0 for it, counted by BOTH")
gt3 = np.zeros((H, W), np.int64)
gt3[:, :3] = 1
pred3 = gt3.copy()
pred3[10:14, 10:14] = 5                   # predict class 5, which never appears in GT
pm, om = project_miou(pred3, gt3), official_miou(pred3, gt3)
check("project == official (absent-but-predicted class)", abs(pm - om) < TOL, f"proj={pm:.12f} off={om:.12f}")

# ---------------------------------------------------------------------------
print("\n[4] fuzz: 30 random scenes, varying error rate and ignore fraction")
ok, worst = True, 0.0
for s in range(30):
    r = np.random.default_rng(100 + s)
    g = r.integers(0, 19, size=(64, 64)).astype(np.int64)
    for t in range(19):
        g.flat[t] = t
    g[r.random((64, 64)) < 0.15] = 255
    pr = g.copy()
    fl = r.random((64, 64)) < (0.1 + 0.5 * r.random())
    pr[fl] = r.integers(0, 19, size=int(fl.sum()))
    pr[g == 255] = r.integers(0, 19, size=int((g == 255).sum()))
    a, b = project_miou(pr, g), official_miou(pr, g)
    worst = max(worst, abs(a - b))
    ok = ok and abs(a - b) < TOL
check("all 30 random scenes bit-match the official routine", ok, f"worst |d|={worst:.2e}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
n = sum(results)
print(f"RESULT: {n}/{len(results)} checks passed — "
      f"{'project mIoU IS the official Cityscapes mIoU' if n == len(results) else 'MISMATCH vs official'}")
sys.exit(0 if n == len(results) else 1)

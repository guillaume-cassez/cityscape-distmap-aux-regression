#!/usr/bin/env python3
"""
Re-evaluate trained checkpoints on the Cityscapes val set with OFFICIAL metrics
and the CORRECTED contour estimators (v0.2.0 erratum).

For each checkpoint:
  * mIoU + per-class IoU  -> computed by the OFFICIAL cityscapesScripts routine
    (cityscapesscripts.evaluation.getIouScoreForLabel + getScoreAverage) applied to
    the real val confusion matrix, and cross-checked bit-for-bit against
    src.metrics.SegmentationMetrics (the equivalence is proven in
    tests/test_official_miou.py). mIoU is the only official Cityscapes ranking metric.
  * Boundary F1 / Trimap IoU -> the CORRECTED per-class estimators
    (per-class binary masks / all inter-class boundaries, not road-vs-rest).

Writes <ckpt>.official.json next to each checkpoint and one aggregate CSV
(default results/pilot_results_official.csv) with columns:
    variant,seed,epoch,mIoU,boundary_f1_mean,trimap_mIoU_mean,<19 per-class official IoU>

Run on the GPU host, with the corrected src/ synced and cityscapesscripts installed:
    CITYSCAPES_ROOT=/home/ser/datasets/cityscapes \
    taskset -c 0-15 python scripts/reeval_official.py \
        --checkpoints "/path/to/checkpoints/*/epoch_160.pth" \
        --out results/pilot_results_official.csv
"""
import argparse
import csv
import glob
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.models import build_model
from src.data import build_dataloaders
from src.metrics import SegmentationMetrics
from src.metrics.segmentation_metrics import compute_boundary_f1, compute_trimap_iou
from src.data.cityscapes_dataset import CLASS_NAMES

import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cseval
from cityscapesscripts.helpers.labels import trainId2label

# trainId 0..18 -> the official (non-ignoreInEval) labelId
TRAINID2ID = {t: trainId2label[t].id for t in range(19)}


def official_from_trainid_cm(cm19: np.ndarray):
    """Official mIoU + per-class IoU from a 19x19 trainId confusion matrix cm19[gt, pred],
    using the official cityscapesScripts routine in labelId space."""
    args = argparse.Namespace()
    conf = cseval.generateMatrix(args)  # labelId-space zeros; also fills args.evalLabels
    for g in range(19):
        for p in range(19):
            conf[TRAINID2ID[g], TRAINID2ID[p]] = cm19[g, p]
    per = {CLASS_NAMES[t]: cseval.getIouScoreForLabel(TRAINID2ID[t], conf, args) for t in range(19)}
    miou = cseval.getScoreAverage(per, args)
    return float(miou), per


def parse_variant_seed(path_str: str):
    """Parse ('A', 42) from a full checkpoint path; works for the original layout
    (.../pilot_fullres_A_ce_seed42/epoch_160.pth), the slimmed layout
    (.../reeval_slim/pilot_fullres_A_ce_seed42__epoch_160.pth), and the paper1 DistMap
    variants Cp=DMdice (CE+Dice+SDT) / Dp=DM (CE+SDT). Check the distmap names FIRST:
    `_([ABCD])_` would miss them, and Cp/Dp must be distinct (no '?' collision)."""
    s = re.search(r"seed(\d+)", path_str)
    seed = int(s.group(1)) if s else -1
    if "DMdice_distmap" in path_str:
        return "Cp", seed
    if "DM_distmap" in path_str:
        return "Dp", seed
    v = re.search(r"_([ABCD])_", path_str)
    return (v.group(1) if v else "?"), seed


def evaluate_ckpt(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    OmegaConf.set_struct(cfg, False)
    # We overwrite every weight via load_state_dict, so skip the pretrained download.
    try:
        cfg.model.backbone.pretrained = "none"
    except Exception:
        pass

    set_seed(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    channels_last = bool(cfg.training.get("channels_last", True))
    model = build_model(cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = build_dataloaders(cfg)["val"]
    metrics = SegmentationMetrics(num_classes=19)
    bf1s, tris = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(images)
                if isinstance(out, tuple):
                    out = out[0]
            metrics.update(out, labels)
            pred = out.argmax(dim=1).cpu().numpy()
            lab = labels.cpu().numpy()
            for b in range(pred.shape[0]):
                bf1s.append(compute_boundary_f1(pred[b], lab[b])["boundary_f1"])
                tris.append(compute_trimap_iou(pred[b], lab[b])["trimap_mIoU"])

    proj = metrics.compute()
    off_miou, off_per = official_from_trainid_cm(metrics.confusion_matrix)
    variant, seed = parse_variant_seed(str(ckpt_path))
    return {
        "checkpoint": str(ckpt_path),
        "run": f"{variant}_seed{seed}",
        "variant": variant,
        "seed": seed,
        "epoch": int(ckpt.get("epoch", -1)) + 1,
        "mIoU_official": off_miou,
        "mIoU_project": float(proj["mIoU"]),
        "mIoU_abs_diff": abs(off_miou - float(proj["mIoU"])),
        "boundary_f1_mean": float(np.mean(bf1s)),
        "trimap_mIoU_mean": float(np.mean(tris)),
        "n_val_images": len(bf1s),
        "per_class_iou_official": {k: (None if math.isnan(v) else float(v)) for k, v in off_per.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, help="glob, e.g. '/.../checkpoints/*/epoch_160.pth'")
    ap.add_argument("--out", default="results/pilot_results_official.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(args.checkpoints))
    if not ckpts:
        sys.exit(f"no checkpoints match: {args.checkpoints}")
    print(f"device={device}  {len(ckpts)} checkpoints")

    rows = []
    for i, c in enumerate(ckpts):
        t0 = time.time()
        rec = evaluate_ckpt(c, device)
        Path(c).with_suffix(".official.json").write_text(json.dumps(rec, indent=2))
        rows.append(rec)
        print(f"[{i+1}/{len(ckpts)}] {rec['run']} e{rec['epoch']}  "
              f"mIoU_off={rec['mIoU_official']*100:.2f} proj={rec['mIoU_project']*100:.2f} "
              f"|d|={rec['mIoU_abs_diff']:.2e}  bf1={rec['boundary_f1_mean']*100:.2f} "
              f"tri={rec['trimap_mIoU_mean']*100:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "seed", "epoch", "mIoU", "boundary_f1_mean", "trimap_mIoU_mean"] + CLASS_NAMES
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rows, key=lambda x: (x["variant"], x["seed"])):
            pc = r["per_class_iou_official"]
            w.writerow(
                [r["variant"], r["seed"], r["epoch"],
                 f"{r['mIoU_official']:.6f}", f"{r['boundary_f1_mean']:.6f}", f"{r['trimap_mIoU_mean']:.6f}"]
                + [f"{(pc[c] if pc[c] is not None else float('nan')):.6f}" for c in CLASS_NAMES]
            )
    print(f"-> {out}")
    bad = [r["run"] for r in rows if r["mIoU_abs_diff"] >= 1e-6]
    print("official-vs-project mIoU mismatch:", bad if bad else "none (all bit-match the official routine)")


if __name__ == "__main__":
    main()

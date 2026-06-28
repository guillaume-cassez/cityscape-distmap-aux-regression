#!/usr/bin/env python3
"""
Consensus evaluation on the Cityscapes val set.

Runs the trained checkpoints, collects per-seed predictions, and evaluates the
two consensus modes from src/postprocessing/consensus.py:

  multiseed : per-pixel majority vote over the 3 seeds of one variant, plus the
              mean agreement (uncertainty) map and fragment count.
  pair      : variant-pair CC veto — `--primary D` vetoed by `--veto B`, per seed,
              reassigning dropped fragments to the veto label.

For each mode it reports mIoU / Boundary F1 / Trimap IoU (corrected per-class
metrics) and the per-class fragment count, against the single-model baselines.

This script needs the GPU and the checkpoints — it is meant to run on the Tower.
The checkpoint root defaults to $CITYSCAPE_CKPT_ROOT, else checkpoints/.

Examples:
    # majority vote over the 3 seeds of variant D
    taskset -c 0-15 python scripts/evaluate_consensus.py --mode multiseed --primary D

    # D vetoed by B (per seed), fragments reassigned to B
    taskset -c 0-15 python scripts/evaluate_consensus.py --mode pair --primary D --veto B
"""
import argparse
import json
import os
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
from src.postprocessing.consensus import (
    cc_veto, majority_vote, agreement_map, count_fragments,
)

PREFIX = {
    "A": "pilot_fullres_A_ce",
    "B": "pilot_fullres_B_baseline",
    "C": "pilot_fullres_C_boundary",
    "D": "pilot_fullres_D_ce_boundary",
    # paper1 DistMap variants (SDT aux head): Cp = CE+Dice+SDT (canonical, BRATS-faithful
    # DistMap), Dp = CE+SDT (no Dice). Canonical paper1 consensus = Cp veto B (Cp⊘B).
    "Cp": "pilot_fullres_DMdice_distmap",
    "Dp": "pilot_fullres_DM_distmap",
}
SEEDS = [42, 123, 456]
NUM_CLASSES = 19


def per_image_cm(pred: np.ndarray, gt: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Per-image 19x19 confusion cm[gt, pred] with the same void-masking as SegmentationMetrics."""
    k = (gt != ignore_index) & (gt >= 0) & (gt < NUM_CLASSES)
    idx = NUM_CLASSES * gt[k].astype(np.int64) + pred[k].astype(np.int64)
    return np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES).astype(np.int64)


def ckpt_root() -> Path:
    return Path(os.environ.get("CITYSCAPE_CKPT_ROOT", "checkpoints"))


def load_model(variant, seed, device):
    path = ckpt_root() / f"{PREFIX[variant]}_seed{seed}" / "epoch_160.pth"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_all(model, loader, device):
    """Return list of (pred_uint8 HxW, label HxW) over the whole loader."""
    out = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last)
        labels = batch["label"].cpu().numpy()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
        preds = logits.argmax(1).cpu().numpy().astype(np.uint8)
        for b in range(preds.shape[0]):
            out.append((preds[b], labels[b]))
    return out


_LEAN = {"on": False}  # set from --lean: skip the heavy per-image Boundary F1 / Trimap loop


def score(pred_label_pairs):
    """mIoU + fragment count over (pred,label) pairs; also Boundary F1 / Trimap unless --lean.

    The consensus claims (§5.5) are Delta mIoU + Delta fragment count + agreement, so --lean
    drops the expensive contour metrics and keeps just what the consensus section reports.
    """
    m = SegmentationMetrics(num_classes=NUM_CLASSES)
    bf1s, tris, frags = [], [], []
    for pred, label in pred_label_pairs:
        m.update(torch.from_numpy(pred[None].astype(np.int64)),
                 torch.from_numpy(label[None].astype(np.int64)))
        if not _LEAN["on"]:
            bf1s.append(compute_boundary_f1(pred, label)["boundary_f1"])
            tris.append(compute_trimap_iou(pred, label)["trimap_mIoU"])
        frags.append(sum(count_fragments(pred).values()))
    res = m.compute()
    out = {"mIoU": res["mIoU"], "fragments_mean": float(np.mean(frags))}
    if not _LEAN["on"]:
        out["boundary_f1_mean"] = float(np.mean(bf1s))
        out["trimap_mIoU_mean"] = float(np.mean(tris))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["multiseed", "pair"], required=True)
    ap.add_argument("--primary", required=True, choices=list(PREFIX))
    ap.add_argument("--veto", choices=list(PREFIX), help="veto variant (pair mode)")
    ap.add_argument("--max-drop-size", type=int, default=None,
                    help="pair mode: only drop fragments up to N px (default: any)")
    ap.add_argument("--out", default="results/consensus_results.json")
    ap.add_argument("--lean", action="store_true",
                    help="skip Boundary F1 / Trimap; keep mIoU + fragments (+ agreement) — ~10x faster")
    ap.add_argument("--dump-cm", default=None,
                    help="pair mode: also save the per-image confusion of the FUSED prediction to "
                         "this dir, for the paired image-bootstrap of the consensus Delta mIoU "
                         "(additive; off by default — core scoring is unchanged)")
    args = ap.parse_args()
    _LEAN["on"] = args.lean

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    torch.backends.cudnn.benchmark = True

    # Build a dataloader once (val split). cfg comes from the primary's seed-42 ckpt.
    _, cfg0 = load_model(args.primary, SEEDS[0], device)
    loader = build_dataloaders(cfg0)["val"]
    labels_ref = None
    t0 = time.time()
    record = {"mode": args.mode, "primary": args.primary, "veto": args.veto}

    if args.mode == "multiseed":
        # Collect the 3 seeds' predictions, score each, then the majority vote.
        per_seed_preds, baselines = [], []
        for s in SEEDS:
            model, _ = load_model(args.primary, s, device)
            pairs = predict_all(model, loader, device)
            if labels_ref is None:
                labels_ref = [lb for _, lb in pairs]
            per_seed_preds.append([p for p, _ in pairs])
            baselines.append(score(pairs))
            del model
            torch.cuda.empty_cache()

        voted_pairs, agrees = [], []
        for i in range(len(labels_ref)):
            seed_maps = [per_seed_preds[k][i] for k in range(len(SEEDS))]
            voted = majority_vote(seed_maps)
            voted_pairs.append((voted.astype(np.uint8), labels_ref[i]))
            agrees.append(float(agreement_map(seed_maps).mean()))

        record["baseline_per_seed"] = baselines
        record["baseline_mean"] = {
            k: float(np.mean([b[k] for b in baselines])) for k in baselines[0]}
        record["consensus"] = score(voted_pairs)
        record["mean_agreement"] = float(np.mean(agrees))

    else:  # pair veto
        assert args.veto, "--veto required in pair mode"
        per_seed = []
        for s in SEEDS:
            mp, _ = load_model(args.primary, s, device)
            prim = predict_all(mp, loader, device)
            del mp; torch.cuda.empty_cache()
            mv, _ = load_model(args.veto, s, device)
            veto = predict_all(mv, loader, device)
            del mv; torch.cuda.empty_cache()
            if labels_ref is None:
                labels_ref = [lb for _, lb in prim]
            fused, fcms = [], []
            for i in range(len(prim)):
                f = cc_veto(prim[i][0].astype(np.int64), veto[i][0].astype(np.int64),
                            max_drop_size=args.max_drop_size)
                fused.append((f.astype(np.uint8), labels_ref[i]))
                if args.dump_cm:
                    fcms.append(per_image_cm(f, labels_ref[i]))
            if args.dump_cm:
                try:
                    d = Path(args.dump_cm)
                    d.mkdir(parents=True, exist_ok=True)
                    np.save(d / f"fused_{args.primary}veto{args.veto}_seed{s}__epoch_160.cm.npy",
                            np.stack(fcms, 0))
                except Exception as e:  # never let the cm dump break the main scoring
                    print(f"  [dump-cm] save failed seed {s}: {e}", flush=True)
            per_seed.append({
                "primary": score(prim),
                "veto": score(veto),
                "fused": score(fused),
            })
        record["per_seed"] = per_seed
        record["fused_mean"] = {
            k: float(np.mean([ps["fused"][k] for ps in per_seed]))
            for k in per_seed[0]["fused"]}

    record["elapsed_s"] = round(time.time() - t0, 1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()

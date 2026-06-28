#!/usr/bin/env python3
"""Paired image-bootstrap of dataset-level mIoU differences — the SPEC primary significance test.

Loads the per-image confusion matrices from dump_perimage_cm.py and, for every variant pair,
resamples the N val images with replacement (PAIRED: the same resampled indices for both
variants), recomputes the dataset-level Delta mIoU per replicate, and reports:
  * each variant's mIoU (seed-mean) with a 95% image-bootstrap CI (percentiles 2.5 / 97.5);
  * each pair's Delta mIoU, 95% CI, and two-sided p = 2*min(frac Delta<=0, frac Delta>=0).

Seeds are AVERAGED within each replicate (the headline statistic is the seed-mean dataset mIoU);
seed variance is the separate Student-t robustness analysis, not this test.

CPU only, numpy only. mIoU uses the project convention, which is bit-identical to the official
cityscapesscripts routine (tests/test_official_miou.py), so cityscapesscripts is not needed here.

    python3 scripts/bootstrap_miou.py --cm-dir results/perimage_cm \
        --variants A,B,C,D --seeds 42,123,456 --B 10000 --out analysis/bootstrap_miou.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

NC = 19


def miou_from_cm(cm: np.ndarray) -> float:
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    valid = union > 0
    iou = np.divide(inter, union, out=np.zeros(NC), where=valid)
    return float(iou[valid].mean()) if valid.any() else 0.0


def load_stack(cm_dir: str, variant: str, seeds) -> np.ndarray:
    """Return (S, N, 19, 19) stacked over seeds for one variant."""
    arrs = []
    for s in seeds:
        p = Path(cm_dir) / f"{variant}_seed{s}__epoch_160.cm.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        arrs.append(np.load(p))
    n = arrs[0].shape[0]
    for a in arrs:
        if a.shape != (n, NC, NC):
            raise ValueError(f"{variant}: shape {a.shape} != {(n, NC, NC)}")
    return np.stack(arrs, 0)


def seedmean_miou(stack_sncc: np.ndarray, idx: np.ndarray) -> float:
    """Seed-mean dataset mIoU over the resampled image indices `idx`."""
    sub = stack_sncc[:, idx].sum(axis=1)  # (S, 19, 19)
    return float(np.mean([miou_from_cm(sub[s]) for s in range(sub.shape[0])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm-dir", default="results/perimage_cm")
    ap.add_argument("--variants", default="A,B,C,D")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260618)
    ap.add_argument("--out", default="analysis/bootstrap_miou.json")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    stacks = {v: load_stack(args.cm_dir, v, seeds) for v in variants}
    n = next(iter(stacks.values())).shape[1]
    rng = np.random.default_rng(args.bootstrap_seed)

    full = np.arange(n)
    point = {v: seedmean_miou(stacks[v], full) for v in variants}

    # shared resample indices across all variants -> the pairwise differences are PAIRED
    boot_idx = [rng.integers(0, n, n) for _ in range(args.B)]
    dist = {v: np.array([seedmean_miou(stacks[v], idx) for idx in boot_idx]) for v in variants}

    out = {
        "test": "paired image-bootstrap (dataset-level mIoU, seed-mean), two-sided",
        "n_val": int(n), "B": int(args.B), "seeds": seeds, "variants": variants,
        "point_miou": {v: point[v] for v in variants},
        "miou_ci95": {v: [float(np.percentile(dist[v], 2.5)),
                          float(np.percentile(dist[v], 97.5))] for v in variants},
        "pairwise": {},
    }
    for x, y in itertools.combinations(variants, 2):
        d = dist[x] - dist[y]
        p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        out["pairwise"][f"{x}_vs_{y}"] = {
            "delta_miou": float(point[x] - point[y]),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(p, 1.0)),
        }

    # Holm correction across the pairwise family (same convention as aggregate_official.py)
    order = sorted(out["pairwise"], key=lambda k: out["pairwise"][k]["p_two_sided"])
    m, run = len(order), 0.0
    for i, k in enumerate(order):
        run = max(run, out["pairwise"][k]["p_two_sided"] * (m - i))
        out["pairwise"][k]["p_holm"] = float(min(run, 1.0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"n_val={n} B={args.B} seeds={seeds}")
    for v in variants:
        lo, hi = out["miou_ci95"][v]
        print(f"  {v}: mIoU={point[v]*100:.2f}  CI95=[{lo*100:.2f}, {hi*100:.2f}]")
    for k, r in out["pairwise"].items():
        lo, hi = r["ci95"]
        sig = "*" if r["p_holm"] < 0.05 else " "
        print(f"  {k}: Δ={r['delta_miou']*100:+.2f}  CI95=[{lo*100:+.2f}, {hi*100:+.2f}]  "
              f"p={r['p_two_sided']:.3f}  Holm={r['p_holm']:.3f} {sig}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

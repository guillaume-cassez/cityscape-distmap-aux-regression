#!/usr/bin/env python3
"""
Aggregate the per-variant official re-eval CSVs (results/official_[ABCD].csv,
each: variant,seed,epoch,mIoU,boundary_f1_mean,trimap_mIoU_mean,<19 class IoU>)
into the paper's tables, with OFFICIAL mIoU and CORRECTED contour metrics.

Outputs (into --out-dir):
  pilot_results_official.csv  - merged 12-row table (fraction units, like the legacy file)
  table_final.csv             - mean +/- 95% Student-t CI at epoch 160 (percent)
  table_significance.csv      - paired-by-seed t-tests, all 6 pairs x 3 metrics,
                                 with Holm-adjusted p within each metric family
  table_perclass.csv          - per-class IoU mean/std + deltas (percent)

Also prints a legacy-vs-corrected comparison for the global table so the shift
from the buggy road-vs-rest estimator is explicit.

Stats: n=3 seeds -> 95% CI uses t_{0.975,df=2}=4.303 (NOT 1.96); significance is a
paired t-test blocking on seed; Holm correction is applied across the 6 pairwise
tests within each metric (controls family-wise error at alpha=0.05).
"""
import argparse
import csv
import glob
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
VARIANTS = [("A", "CE"), ("B", "CE+Dice"), ("C", "CE+Dice+Bnd"), ("D", "CE+Bnd")]
SEEDS = ["42", "123", "456"]
METRICS = ["mIoU", "boundary_f1_mean", "trimap_mIoU_mean"]
# Legacy v0.1.0 central values (road-vs-rest estimator) for the before/after print.
LEGACY = {  # variant: (mIoU%, bnd_f1%, trimap%)
    "A": (81.28, 58.45, 47.83), "B": (81.09, 58.63, 49.02),
    "C": (81.23, 58.53, 48.93), "D": (81.69, 58.67, 47.93),
}


def load_rows(in_dir):
    rows = []
    for f in sorted(glob.glob(str(Path(in_dir) / "official_*.csv"))):
        rows += list(csv.DictReader(open(f)))
    if not rows:
        raise SystemExit(f"no official_*.csv found in {in_dir}")
    return rows


def table(rows):
    """(variant, metric) -> np.array over SEEDS, in percent."""
    t = {}
    for v, _ in VARIANTS:
        for m in METRICS:
            d = {r["seed"]: float(r[m]) for r in rows if r["variant"] == v}
            t[(v, m)] = np.array([d[s] for s in SEEDS]) * 100.0
    return t


def per_class(rows):
    pc = {}
    for v, _ in VARIANTS:
        arr = []
        for s in SEEDS:
            r = next(r for r in rows if r["variant"] == v and r["seed"] == s)
            arr.append([float(r[c]) for c in CLASS_NAMES])
        pc[v] = np.array(arr) * 100.0  # (3 seeds, 19 classes)
    return pc


def ci_t(a):
    return stats.t.ppf(0.975, df=len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))


def holm(pairs):
    """pairs: list of (key, p). Returns {key: p_adj} via Holm step-down."""
    order = sorted(pairs, key=lambda kv: kv[1])
    m = len(order)
    adj, prev = {}, 0.0
    for i, (k, p) in enumerate(order):
        val = min(1.0, (m - i) * p)
        prev = max(prev, val)
        adj[k] = prev
    return adj


def write_merged(rows, out):
    cols = ["variant", "seed", "epoch", "mIoU", "boundary_f1_mean", "trimap_mIoU_mean"] + CLASS_NAMES
    with open(out / "pilot_results_official.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for v, _ in VARIANTS:
            for s in SEEDS:
                r = next(r for r in rows if r["variant"] == v and r["seed"] == s)
                w.writerow([r.get(c, "") for c in cols])
    print(f"-> pilot_results_official.csv")


def write_final(t, out):
    lines = ["variant,name,mIoU_mean,mIoU_ci,bnd_f1_mean,bnd_f1_ci,trimap_mean,trimap_ci"]
    print("\n=== table_final (mean +/- 95% Student-t CI) — LEGACY vs CORRECTED ===")
    print(f"{'var':3} {'mIoU':>14} {'BoundaryF1 (legacy->corr)':>30} {'Trimap (legacy->corr)':>28}")
    for v, name in VARIANTS:
        cells = [v, name]
        vals = {}
        for m in METRICS:
            a = t[(v, m)]
            cells += [f"{a.mean():.2f}", f"{ci_t(a):.2f}"]
            vals[m] = (a.mean(), ci_t(a))
        lines.append(",".join(cells))
        lm, lb, lt = LEGACY[v]
        print(f"{v:3} {vals['mIoU'][0]:6.2f}+-{vals['mIoU'][1]:.2f}   "
              f"BF1 {lb:5.2f}->{vals['boundary_f1_mean'][0]:5.2f}+-{vals['boundary_f1_mean'][1]:.2f}   "
              f"Tri {lt:5.2f}->{vals['trimap_mIoU_mean'][0]:5.2f}+-{vals['trimap_mIoU_mean'][1]:.2f}")
    (out / "table_final.csv").write_text("\n".join(lines) + "\n")
    print("-> table_final.csv")


def write_significance(t, out):
    lines = ["comparison,metric,delta_mean,t_stat,p_value,p_holm,significant_holm_0.05"]
    print("\n=== table_significance (paired t-test by seed, Holm-adjusted within metric) ===")
    for m in METRICS:
        raw = []
        for (a, _), (b, _) in combinations(VARIANTS, 2):
            x, y = t[(a, m)], t[(b, m)]
            ts, p = stats.ttest_rel(x, y)
            raw.append((f"{a}-{b}", float(p), float(ts), float((x - y).mean())))
        adj = holm([(k, p) for k, p, _, _ in raw])
        for k, p, ts, d in raw:
            pa = adj[k]
            sig = "yes" if pa < 0.05 else "no"
            lines.append(f"{k},{m},{d:+.3f},{ts:.3f},{p:.4f},{pa:.4f},{sig}")
            print(f"  {k:4} {m:18} d={d:+6.3f} t={ts:7.3f} p={p:.4f} p_holm={pa:.4f} {sig}")
    (out / "table_significance.csv").write_text("\n".join(lines) + "\n")
    print("-> table_significance.csv")


def write_perclass(pc, out):
    means = {v: pc[v].mean(axis=0) for v, _ in VARIANTS}
    head = "class," + ",".join(f"{v}_mean,{v}_std" for v, _ in VARIANTS) + ",delta_D_A,delta_D_B,delta_C_B"
    lines = [head]
    for ci, c in enumerate(CLASS_NAMES):
        row = [c]
        for v, _ in VARIANTS:
            row += [f"{pc[v][:, ci].mean():.2f}", f"{pc[v][:, ci].std(ddof=1):.2f}"]
        row += [f"{means['D'][ci]-means['A'][ci]:+.2f}",
                f"{means['D'][ci]-means['B'][ci]:+.2f}",
                f"{means['C'][ci]-means['B'][ci]:+.2f}"]
        lines.append(",".join(row))
    (out / "table_perclass.csv").write_text("\n".join(lines) + "\n")
    print("-> table_perclass.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results", help="dir with official_[ABCD].csv")
    ap.add_argument("--out-dir", default="papers/paper2/figures")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.in_dir)
    print(f"loaded {len(rows)} rows")
    t = table(rows)
    write_merged(rows, out)
    write_final(t, out)
    write_significance(t, out)
    write_perclass(per_class(rows), out)


if __name__ == "__main__":
    main()

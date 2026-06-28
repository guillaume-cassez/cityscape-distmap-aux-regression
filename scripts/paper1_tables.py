#!/usr/bin/env python3
"""Paper1 (DistMap) paper tables — the analogue of scripts/aggregate_official.py for paper2,
adapted to the variant set A=CE, B=CE+Dice, C′=CE+Dice+DistMap(SDT), D′=CE+DistMap(SDT).

A/B per-seed metrics come from paper2's official per-seed CSV (shared, correctly labelled);
C′/D′ come from results/paper1_official.csv whose variant column is '?', so each row is mapped
to C′(Cp)/D′(Dp) by nearest per-seed mIoU derived from the (correctly named) confusion matrices.

Outputs into --out-dir (default papers/paper1/figures):
  table_final.csv        - mean +/- 95% Student-t CI at epoch 160 (percent), t(.975,df=2)=4.303
  table_significance.csv - paired-by-seed t-tests, 6 pairs x 3 metrics, Holm within metric
  table_perclass.csv     - per-class IoU mean/std + deltas (percent)
  pilot_results_official.csv - merged 12-row per-seed table (fraction units)

    python3 scripts/paper1_tables.py --out-dir papers/paper1/figures
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

NC = 19
SEEDS = ["42", "123", "456"]
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
# (key, label, loss-name)
VARIANTS = [("A", "A", "CE"), ("B", "B", "CE+Dice"),
            ("Cp", "C′", "CE+Dice+DistMap"), ("Dp", "D′", "CE+DistMap")]
METRICS = ["mIoU", "boundary_f1_mean", "trimap_mIoU_mean"]
CM_DIR = Path("results/perimage_cm")


def miou_from_cm(cm):
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    valid = union > 0
    iou = np.divide(inter, union, out=np.zeros(NC), where=valid)
    return float(iou[valid].mean()) if valid.any() else 0.0


def cm_miou(variant, seed):
    return miou_from_cm(np.load(CM_DIR / f"{variant}_seed{seed}__epoch_160.cm.npy").sum(0))


def load_rows():
    """Return {key: {seed: dict-row}} with correctly-labelled A/B/Cp/Dp."""
    out = {k: {} for k, _, _ in VARIANTS}
    for r in csv.DictReader(open("papers/paper2/publish/repo/data/pilot_results_official.csv")):
        if r["variant"] in ("A", "B"):
            out[r["variant"]][r["seed"]] = r
    ref = {k: {s: cm_miou(k, int(s)) for s in SEEDS} for k in ("Cp", "Dp")}
    seen = {s: set() for s in SEEDS}  # per-seed uniqueness: a global 3/3 count misses an
    for r in csv.DictReader(open("results/paper1_official.csv")):  # intra-seed collision
        s, mi = r["seed"], float(r["mIoU"])
        k = min(("Cp", "Dp"), key=lambda v: abs(mi - ref[v][s]))
        assert k not in seen[s], f"C'/D' mapping collision: seed {s} matched {k} twice"
        seen[s].add(k)
        out[k][s] = r
    assert all(seen[s] == {"Cp", "Dp"} for s in SEEDS), f"C'/D' mapping incomplete: {seen}"
    return out


def table(rows):
    t = {}
    for k, _, _ in VARIANTS:
        for m in METRICS:
            t[(k, m)] = np.array([float(rows[k][s][m]) for s in SEEDS]) * 100.0
    return t


def per_class(rows):
    return {k: np.array([[float(rows[k][s][c]) for c in CLASS_NAMES] for s in SEEDS]) * 100.0
            for k, _, _ in VARIANTS}


def ci_t(a):
    return float(stats.t.ppf(0.975, df=len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


def holm(pairs):
    order = sorted(pairs, key=lambda kv: kv[1])
    m, prev, adj = len(order), 0.0, {}
    for i, (k, p) in enumerate(order):
        prev = max(prev, min(1.0, (m - i) * p))
        adj[k] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="papers/paper1/figures")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    t = table(rows)
    pc = per_class(rows)

    # merged per-seed CSV (fraction units, like the legacy file)
    cols = ["variant", "seed", "epoch", "mIoU", "boundary_f1_mean", "trimap_mIoU_mean"] + CLASS_NAMES
    with open(out / "pilot_results_official.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k, label, _ in VARIANTS:
            for s in SEEDS:
                r = dict(rows[k][s]); r["variant"] = label
                w.writerow([r.get(c, "") for c in cols])

    # table_final
    lines = ["variant,name,mIoU_mean,mIoU_ci,bnd_f1_mean,bnd_f1_ci,trimap_mean,trimap_ci"]
    for k, label, name in VARIANTS:
        cells = [label, name]
        for m in METRICS:
            a = t[(k, m)]
            cells += [f"{a.mean():.2f}", f"{ci_t(a):.2f}"]
        lines.append(",".join(cells))
    (out / "table_final.csv").write_text("\n".join(lines) + "\n")

    # table_significance (paired-t by seed, Holm within metric)
    lines = ["comparison,metric,delta_mean,t_stat,p_value,p_holm,significant_holm_0.05"]
    lab = {k: label for k, label, _ in VARIANTS}
    for m in METRICS:
        raw = []
        for (a, _, _), (b, _, _) in combinations(VARIANTS, 2):
            x, y = t[(a, m)], t[(b, m)]
            ts, p = stats.ttest_rel(x, y)
            raw.append((f"{lab[a]}-{lab[b]}", float(p), float(ts), float((x - y).mean())))
        adj = holm([(k, p) for k, p, _, _ in raw])
        for k, p, ts, d in raw:
            sig = "yes" if adj[k] < 0.05 else "no"
            lines.append(f"{k},{m},{d:+.3f},{ts:.3f},{p:.4f},{adj[k]:.4f},{sig}")
    (out / "table_significance.csv").write_text("\n".join(lines) + "\n")

    # table_perclass
    means = {k: pc[k].mean(axis=0) for k, _, _ in VARIANTS}
    head = "class," + ",".join(f"{lab[k]}_mean,{lab[k]}_std" for k, _, _ in VARIANTS)
    head += ",delta_Dp_A,delta_Dp_B,delta_Cp_B"
    lines = [head]
    for ci, c in enumerate(CLASS_NAMES):
        row = [c]
        for k, _, _ in VARIANTS:
            row += [f"{pc[k][:, ci].mean():.2f}", f"{pc[k][:, ci].std(ddof=1):.2f}"]
        row += [f"{means['Dp'][ci]-means['A'][ci]:+.2f}",
                f"{means['Dp'][ci]-means['B'][ci]:+.2f}",
                f"{means['Cp'][ci]-means['B'][ci]:+.2f}"]
        lines.append(",".join(row))
    (out / "table_perclass.csv").write_text("\n".join(lines) + "\n")

    for fn in ("table_final.csv", "table_significance.csv", "table_perclass.csv",
               "pilot_results_official.csv"):
        print(f"-> {out / fn}")


if __name__ == "__main__":
    main()

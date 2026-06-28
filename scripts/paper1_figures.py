#!/usr/bin/env python3
"""Paper1 (DistMap) figures. Self-contained: reads the already-aggregated
papers/paper1/figures/table_perclass.csv (produced by scripts/paper1_tables.py),
so no checkpoints / GPU are needed.

Output (--out-dir, default papers/paper1/figures):
  fig_perclass.png / .pdf  - per-class IoU at epoch 160, variants A/B/C′/D′,
                             sorted by Δ(D′−B) descending (mirrors paper2 fig_perclass).

The convergence figure of paper2 is intentionally NOT reproduced: paper1's intermediate
val curves are unavailable (checkpoints purged), so convergence is reported as the
epoch-10→160 two-point crossover in results/paper1_numbers.md §1b instead of a fake curve.

    python3 scripts/paper1_figures.py
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (column-prefix in table_perclass.csv, legend label, colour) — mirrors paper2 palette
VARIANTS = [("A", "A: CE", "#888888"), ("B", "B: CE+Dice", "#1f77b4"),
            ("C′", "C′: CE+Dice+DistMap", "#2ca02c"), ("D′", "D′: CE+DistMap", "#d62728")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="papers/paper1/figures")
    args = ap.parse_args()
    out = Path(args.out_dir)
    rows = list(csv.DictReader(open(out / "table_perclass.csv")))
    # sort classes by Δ(D′−B) descending (the no-Dice-vs-baseline contrast)
    rows.sort(key=lambda r: float(r["delta_Dp_B"]), reverse=True)
    classes = [r["class"] for r in rows]
    x = np.arange(len(classes))
    w = 0.2

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (key, label, color) in enumerate(VARIANTS):
        vals = [float(r[f"{key}_mean"]) for r in rows]
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(50, 100)
    ax.set_title("Per-class IoU at epoch 160 (3-seed mean), sorted by Δ(D′−B)")
    ax.legend(ncol=4, loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_perclass.{ext}", dpi=150)
        print(f"-> {out / f'fig_perclass.{ext}'}")
    plt.close(fig)


if __name__ == "__main__":
    main()

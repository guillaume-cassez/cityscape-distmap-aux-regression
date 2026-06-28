#!/usr/bin/env python3
"""Consensus ablation MATRIX — evaluate several (primary ⊘ veto) pairings in ONE inference pass.

Instead of arguing which consensus construction is right, MEASURE them. This caches each variant's
val predictions once per seed, then applies the CC-veto for every requested pairing on CPU — so N
pairings cost the same GPU inference as evaluating the variants once. Reports, per single variant
and per pairing, mIoU + connected-component fragment count (mean over seeds ± Student-t 95% CI),
and each pairing's Δ vs its primary. With --full it also adds Boundary F1 / Trimap (slow).

Lets the numbers decide the canonical consensus (e.g. BRATS-faithful C⊘B vs current D⊘B vs C⊘A …)
and directly tests the thesis link: do the boundary models (C, D) fragment more than the baselines
(A=CE, B=CE+Dice), and does a baseline veto prune that at no mIoU cost?

Run on the GPU host (Tower), alone (one heavy job at a time — full-res dataloader):
    CITYSCAPES_ROOT=/home/ser/datasets/cityscapes CITYSCAPE_CKPT_ROOT=$PWD/reeval_ckpt_layout \
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 python3 -u scripts/evaluate_consensus_matrix.py \
        --pairings C/B,D/B,C/A,D/A --out results/consensus_matrix.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate_consensus as ec  # reuse load_model / predict_all / score / _LEAN / SEEDS
from src.postprocessing.consensus import cc_veto


def tci(xs):
    """mean, Student-t 95% half-width (t(.975,df=n-1)) — 4.303 at n=3."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    m = float(xs.mean())
    if n < 2:
        return m, 0.0
    se = xs.std(ddof=1) / np.sqrt(n)
    return m, float(stats.t.ppf(0.975, df=n - 1) * se)


def agg(per_seed_dicts):
    keys = list(per_seed_dicts[0].keys())
    out = {}
    for k in keys:
        m, h = tci([d[k] for d in per_seed_dicts])
        out[k] = {"mean": m, "ci95": h}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairings", default="C/B,D/B,C/A,D/A",
                    help="comma list of primary/veto, e.g. C/B,D/B,C/A,D/A")
    ap.add_argument("--full", action="store_true", help="also Boundary F1 / Trimap (slow contour loop)")
    ap.add_argument("--max-drop-size", type=int, default=None)
    ap.add_argument("--out", default="results/consensus_matrix.json")
    args = ap.parse_args()
    ec._LEAN["on"] = not args.full

    pairings = [tuple(p.split("/")) for p in args.pairings.split(",")]
    variants = sorted({v for pr in pairings for v in pr})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ec.set_seed(42)
    torch.backends.cudnn.benchmark = True
    print(f"device={device}  variants={variants}  pairings={['/'.join(p) for p in pairings]}  "
          f"lean={not args.full}", flush=True)

    singles = {v: [] for v in variants}
    pairs = {f"{p}/{v}": [] for p, v in pairings}

    for s in ec.SEEDS:
        t0 = time.time()
        preds, labels_ref, loader = {}, None, None
        for v in variants:
            model, cfg = ec.load_model(v, s, device)
            if loader is None:
                loader = ec.build_dataloaders(cfg)["val"]
            pr = ec.predict_all(model, loader, device)
            if labels_ref is None:
                labels_ref = [lb.astype(np.uint8) for _, lb in pr]
            preds[v] = [p.astype(np.uint8) for p, _ in pr]
            del model, pr
            torch.cuda.empty_cache()

        for v in variants:
            singles[v].append(ec.score(list(zip(preds[v], labels_ref))))
        for p, vv in pairings:
            fused = []
            for i in range(len(labels_ref)):
                f = cc_veto(preds[p][i].astype(np.int64), preds[vv][i].astype(np.int64),
                            max_drop_size=args.max_drop_size)
                fused.append((f.astype(np.uint8), labels_ref[i]))
            pairs[f"{p}/{vv}"].append(ec.score(fused))
        del preds
        print(f"  seed {s} done ({time.time()-t0:.0f}s)", flush=True)

    out = {"seeds": ec.SEEDS, "lean": not args.full,
           "singletons": {v: agg(singles[v]) for v in variants},
           "pairings": {}}
    for p, vv in pairings:
        k = f"{p}/{vv}"
        a = agg(pairs[k])
        a["delta_vs_primary"] = {
            "mIoU": a["mIoU"]["mean"] - out["singletons"][p]["mIoU"]["mean"],
            "fragments_mean": a["fragments_mean"]["mean"] - out["singletons"][p]["fragments_mean"]["mean"],
        }
        out["pairings"][k] = a

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\nSINGLE variants (mean ± t95):")
    for v in variants:
        a = out["singletons"][v]
        print(f"  {v}: mIoU={a['mIoU']['mean']*100:.2f}±{a['mIoU']['ci95']*100:.2f}  "
              f"frags/img={a['fragments_mean']['mean']:.1f}±{a['fragments_mean']['ci95']:.1f}")
    print("\nCONSENSUS primary⊘veto:")
    for p, vv in pairings:
        k = f"{p}/{vv}"
        a = out["pairings"][k]
        d = a["delta_vs_primary"]
        base = out["singletons"][p]["fragments_mean"]["mean"]
        pct = 100 * d["fragments_mean"] / base if base else 0.0
        print(f"  {k}: mIoU={a['mIoU']['mean']*100:.2f}  ΔmIoU={d['mIoU']*100:+.3f}pp  "
              f"Δfrags={d['fragments_mean']:+.1f} ({pct:+.1f}%)")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

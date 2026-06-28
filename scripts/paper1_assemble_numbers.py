#!/usr/bin/env python3
"""Assemble results/paper1_numbers.md — the consolidated source of truth for the paper1
(DistMap) manuscript, mirroring results/paper2_v0.2.2_numbers.md section-for-section.

Variants (paper1 namespace): A = CE, B = CE+Dice (= nnU-Net/BRATS baseline), and the two
DistMap (signed distance transform aux head) variants C' = CE+Dice+SDT, D' = CE+SDT.
A/B are shared with paper2; only C'/D' are paper1-specific. Internally the DistMap variants
are keyed "Cp"/"Dp" (the on-disk checkpoint / CM names); they render as C'/D' in the markdown.

Inputs (all already on disk, no GPU):
  * per-image confusion matrices  results/perimage_cm/{A,B,Cp,Dp}_seed{S}__epoch_160.cm.npy
      -> per-seed singleton mIoU (bit-exact official; used to LABEL the '?' rows of the
         paper1 official CSV by nearest-mIoU match, no magic threshold)
  * results/paper1_official.csv                 (per-seed BF1/Trimap for C'/D'; variant='?')
  * papers/paper2/publish/repo/data/pilot_results_official.csv  (per-seed A/B BF1/Trimap)
  * results/paper1_consensus_{CpB,DpB}_full.json   (C'⊘B / D'⊘B, all 4 metrics per seed)
  * results/paper1_consensus_{CpA,DpA}_lean.json   (C'⊘A / D'⊘A matrix rows; optional)
  * analysis/paper1_bootstrap_{variants,CpvetoB,DpvetoB}.json  (primary image-bootstrap)

Stats: primary = paired image-bootstrap (already computed). Secondary = seed paired-t
(n=3, scipy.stats.ttest_rel) + Holm within each metric family, computed here. Both reported.

    python3 scripts/paper1_assemble_numbers.py            # -> results/paper1_numbers.md
"""
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

NC = 19
SEEDS = [42, 123, 456]
CM_DIR = Path("results/perimage_cm")
ROOT = Path(".")
OUT = Path("results/paper1_numbers.md")

# render keys: internal -> manuscript label + loss
LOSS = {"A": "CE", "B": "CE+Dice (= nnU-Net/BRATS baseline)",
        "Cp": "CE+Dice+DistMap (SDT)", "Dp": "CE+DistMap (SDT)"}
LABEL = {"A": "A", "B": "B", "Cp": "C′", "Dp": "D′"}
VARIANTS = ["A", "B", "Cp", "Dp"]
METRICS = ["mIoU", "boundary_f1_mean", "trimap_mIoU_mean"]
MNAME = {"mIoU": "mIoU", "boundary_f1_mean": "Boundary F1", "trimap_mIoU_mean": "Trimap IoU"}


def miou_from_cm(cm):
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    valid = union > 0
    iou = np.divide(inter, union, out=np.zeros(NC), where=valid)
    return float(iou[valid].mean()) if valid.any() else 0.0


def seed_miou_from_cm(variant, seed):
    cm = np.load(CM_DIR / f"{variant}_seed{seed}__epoch_160.cm.npy").sum(0)
    return miou_from_cm(cm)


def holm(items):
    """items: list of (key, p). Return {key: p_holm} step-down."""
    order = sorted(items, key=lambda kv: kv[1])
    m, prev, adj = len(order), 0.0, {}
    for i, (k, p) in enumerate(order):
        prev = max(prev, min(1.0, (m - i) * p))
        adj[k] = prev
    return adj


def ci_t(a):
    a = np.asarray(a, float)
    return float(stats.t.ppf(0.975, df=len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


# ---- 1. per-seed singleton metrics -----------------------------------------
def load_singletons():
    """per[variant][metric][seed] = value in percent (mIoU/BF1/Trimap)."""
    per = {v: {m: {} for m in METRICS} for v in VARIANTS}
    # mIoU (all variants) straight from the CMs — correctly labelled, bit-exact
    for v in VARIANTS:
        for s in SEEDS:
            per[v]["mIoU"][s] = seed_miou_from_cm(v, s) * 100.0
    # BF1/Trimap for A,B from the paper2 official per-seed CSV (correctly labelled)
    p2 = list(csv.DictReader(open(ROOT / "papers/paper2/publish/repo/data/pilot_results_official.csv")))
    for r in p2:
        if r["variant"] in ("A", "B"):
            for m in ("boundary_f1_mean", "trimap_mIoU_mean"):
                per[r["variant"]][m][int(r["seed"])] = float(r[m]) * 100.0
    # BF1/Trimap for C'/D' from paper1_official.csv — variant is '?', so map each row to
    # Cp/Dp by nearest per-seed mIoU (the CM-derived, correctly-labelled value).
    cm_miou = {v: {s: seed_miou_from_cm(v, s) for s in SEEDS} for v in ("Cp", "Dp")}
    rows = list(csv.DictReader(open(ROOT / "results/paper1_official.csv")))
    seen = {s: set() for s in SEEDS}  # per-seed uniqueness: a global 3/3 count misses an
    for r in rows:                    # intra-seed collision compensated across seeds
        s = int(r["seed"])
        mi = float(r["mIoU"])
        v = min(("Cp", "Dp"), key=lambda k: abs(mi - cm_miou[k][s]))
        assert v not in seen[s], f"C'/D' mapping collision: seed {s} matched {v} twice"
        seen[s].add(v)
        for m in ("boundary_f1_mean", "trimap_mIoU_mean"):
            per[v][m][s] = float(r[m]) * 100.0
    assert all(seen[s] == {"Cp", "Dp"} for s in SEEDS), f"C'/D' row mapping incomplete: {seen}"
    return per


# ---- consensus JSON helpers -------------------------------------------------
def cons_full(path):
    """Return dict with per-seed primary/veto/fused arrays (by seed order) + means."""
    r = json.load(open(path))
    ps = r["per_seed"]
    keys = ["mIoU", "fragments_mean", "boundary_f1_mean", "trimap_mIoU_mean"]
    out = {"elapsed_s": r.get("elapsed_s")}
    for which in ("primary", "veto", "fused"):
        out[which] = {k: np.array([sd[which][k] for sd in ps], float) for k in keys}
    return out


def cons_lean_mean(path):
    """Lean ⊘A JSON -> dict of seed-mean primary/veto/fused {mIoU, fragments_mean}."""
    if not Path(path).exists():
        return None
    r = json.load(open(path))
    ps = r["per_seed"]
    return {which: {k: float(np.mean([sd[which][k] for sd in ps]))
                    for k in ("mIoU", "fragments_mean")}
            for which in ("primary", "veto", "fused")}


def boot(path):
    return json.load(open(path)) if Path(path).exists() else None


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    per = load_singletons()
    cb = cons_full("results/paper1_consensus_CpB_full.json")
    db = cons_full("results/paper1_consensus_DpB_full.json")
    ca = cons_lean_mean("results/paper1_consensus_CpA_lean.json")
    da = cons_lean_mean("results/paper1_consensus_DpA_lean.json")
    bv = boot("analysis/paper1_bootstrap_variants.json")
    bcb = boot("analysis/paper1_bootstrap_CpvetoB.json")
    bdb = boot("analysis/paper1_bootstrap_DpvetoB.json")

    # singleton means + fragments/img (B from ⊘B veto, C'/D' from ⊘B primary, A from ⊘A veto/paper2)
    mean = {v: {m: float(np.mean([per[v][m][s] for s in SEEDS])) for m in METRICS} for v in VARIANTS}
    frag = {"B": float(np.mean(cb["veto"]["fragments_mean"])),
            "Cp": float(np.mean(cb["primary"]["fragments_mean"])),
            "Dp": float(np.mean(db["primary"]["fragments_mean"]))}
    frag["A"] = ca["veto"]["fragments_mean"] if ca else (da["veto"]["fragments_mean"] if da else 734.4)
    a_note = "" if (ca or da) else "  *(A from paper2; ⊘A pending)*"

    # seed paired-t over variant pairs, Holm within metric
    sig = {}  # (a,b,metric) -> (delta, t, p, p_holm)
    for m in METRICS:
        raw = []
        for a, b in combinations(VARIANTS, 2):
            x = np.array([per[a][m][s] for s in SEEDS])
            y = np.array([per[b][m][s] for s in SEEDS])
            t, p = stats.ttest_rel(x, y)
            raw.append((f"{a}|{b}", float(p), float(t), float((x - y).mean())))
        adj = holm([(k, p) for k, p, _, _ in raw])
        for k, p, t, d in raw:
            sig[(k, m)] = (d, t, p, adj[k])

    L = []
    L.append("# paper1 v0.1.0 — consolidated numbers (source of truth for the manuscript)")
    L.append("")
    L.append("Val 500 img, 3 seeds (42/123/456), epoch 160. mIoU = official cityscapesscripts "
             "(bit-exact). Variants: A=CE, B=CE+Dice (=nnU-Net/BRATS baseline), C′=CE+Dice+DistMap, "
             "D′=CE+DistMap; DistMap = signed-distance-transform auxiliary regression head. A/B "
             "shared with paper2 (Kervadec). Raw JSONs in results/ and analysis/.")
    L.append("")

    # ---- §1 singletons ----
    L.append("## 1. Variant singletons (mean over 3 seeds)")
    L.append("| | loss | mIoU | Boundary F1 | Trimap IoU | fragments/img |")
    L.append("|---|---|---|---|---|---|")
    best_miou = max(mean[v]["mIoU"] for v in VARIANTS)
    best_bf1 = max(mean[v]["boundary_f1_mean"] for v in VARIANTS)
    for v in VARIANTS:
        mi = f"**{mean[v]['mIoU']:.2f}**" if mean[v]["mIoU"] == best_miou else f"{mean[v]['mIoU']:.2f}"
        bf = (f"**{mean[v]['boundary_f1_mean']:.2f}**"
              if mean[v]["boundary_f1_mean"] == best_bf1 else f"{mean[v]['boundary_f1_mean']:.2f}")
        L.append(f"| {LABEL[v]} | {LOSS[v]} | {mi} | {bf} | "
                 f"{mean[v]['trimap_mIoU_mean']:.2f} | {frag[v]:.1f}{a_note if v=='A' else ''} |")
    L.append("")
    L.append(f"Dice axis: with-Dice (B,C′) ≈ {(frag['B']+frag['Cp'])/2:.0f} fragments/img vs no-Dice "
             f"(A,D′) ≈ {(frag['A']+frag['Dp'])/2:.0f} — the DistMap term itself barely moves fragment "
             f"count (C′−B = {frag['Cp']-frag['B']:+.0f}, D′−A = {frag['Dp']-frag['A']:+.0f}); like "
             "paper2, fragmentation is governed by the Dice axis, not the auxiliary boundary term.")
    L.append("")

    # ---- §1b convergence crossover (epoch 10 -> 160) ----
    ep10 = Path("results/paper1_smallep_official.csv")
    if ep10.exists():
        rows10 = list(csv.DictReader(open(ep10)))
        # phase-1 small-ep = seed 42 only; the 2 '?' rows are C'/D'. The Dice signature
        # (with-Dice -> higher Trimap) is epoch-stable, unlike mIoU which crosses over,
        # so split the pair by Trimap: higher = C' (with-Dice), lower = D'. No magic threshold.
        rows10 = sorted(rows10, key=lambda r: float(r["trimap_mIoU_mean"]))
        d10, c10 = (rows10[0], rows10[-1]) if len(rows10) == 2 else (None, None)
        if c10 and d10:
            c10m, d10m = float(c10["mIoU"]) * 100, float(d10["mIoU"]) * 100
            L.append("## 1b. Convergence crossover (epoch 10 → 160)")
            L.append("Phase-1 small-ep checkpoints are **seed 42 only** (n=1, illustrative); epoch-160 is "
                     "the 3-seed mean. Intermediate val curves are unavailable (checkpoints purged), so "
                     "the crossover is shown as two points, not a continuous curve.")
            L.append("")
            L.append("| variant | mIoU @ epoch 10 (seed 42) | mIoU @ epoch 160 (3-seed) |")
            L.append("|---|---|---|")
            L.append(f"| C′ (CE+Dice+DistMap) | **{c10m:.2f}** | {mean['Cp']['mIoU']:.2f} |")
            L.append(f"| D′ (CE+DistMap) | {d10m:.2f} | **{mean['Dp']['mIoU']:.2f}** |")
            L.append("")
            L.append(f"At epoch 10 C′ leads D′ by {c10m-d10m:+.2f} mIoU (Dice helps early); by epoch 160 "
                     f"D′ overtakes C′ by {mean['Dp']['mIoU']-mean['Cp']['mIoU']:+.2f} — the same "
                     "late-training crossover as paper2 (the no-Dice variant catches up and wins mIoU).")
            L.append("")

    # ---- §2 pairwise mIoU, two tests ----
    L.append("## 2. Variant pairwise mIoU — two tests (report BOTH, no cherry-pick)")
    L.append("**Primary = paired image-bootstrap (n=500, B=10 000, Holm)** — "
             "`analysis/paper1_bootstrap_variants.json`. **Secondary = seed paired-t (n=3, Holm)** "
             "(underpowered, robustness only).")
    L.append("")
    L.append("| pair (X−Y) | Δ mIoU | bootstrap p (Holm) | seed-t p (Holm) |")
    L.append("|---|---|---|---|")
    bvp = bv["pairwise"] if bv else {}
    bkey = {("A", "B"): "A_vs_B", ("A", "Cp"): "A_vs_Cp", ("A", "Dp"): "A_vs_Dp",
            ("B", "Cp"): "B_vs_Cp", ("B", "Dp"): "B_vs_Dp", ("Cp", "Dp"): "Cp_vs_Dp"}
    for a, b in combinations(VARIANTS, 2):
        br = bvp.get(bkey[(a, b)], {})
        d_boot = br.get("delta_miou", 0) * 100
        pb, pbh = br.get("p_two_sided", float("nan")), br.get("p_holm", float("nan"))
        d_seed, _, ps, psh = sig[(f"{a}|{b}", "mIoU")]
        sb = " *" if pbh < 0.05 else ""
        ss = " *" if psh < 0.05 else ""
        L.append(f"| {LABEL[a]}−{LABEL[b]} | {d_boot:+.2f} | {fmt_p(pb)} ({fmt_p(pbh)}){sb} "
                 f"| {fmt_p(ps)} ({fmt_p(psh)}){ss} |")
    L.append("")
    L.append("D′ (CE+DistMap, no Dice) has the highest mIoU and significantly beats B and C′ on the "
             "primary test — the same Dice-axis direction as paper2 (the no-Dice variant wins mIoU "
             "while trading boundary quality, see §4–5).")
    L.append("")

    # ---- §3 consensus matrix (lean) ----
    L.append("## 3. Consensus ablation matrix (mIoU + fragments)")
    L.append("| pairing | mIoU | ΔmIoU | Δfragments |")
    L.append("|---|---|---|---|")

    def matrix_row(name, prim_frag, fused_mi, prim_mi, fused_frag):
        dmi = (fused_mi - prim_mi)
        dfr = 100.0 * (fused_frag - prim_frag) / prim_frag
        L.append(f"| {name} | {fused_mi*100:.2f} | {dmi*100:+.3f} pp | {dfr:+.1f} % |")

    matrix_row("**C′⊘B (canonical)**",
               float(np.mean(cb["primary"]["fragments_mean"])),
               float(np.mean(cb["fused"]["mIoU"])), float(np.mean(cb["primary"]["mIoU"])),
               float(np.mean(cb["fused"]["fragments_mean"])))
    matrix_row("D′⊘B (contrast)",
               float(np.mean(db["primary"]["fragments_mean"])),
               float(np.mean(db["fused"]["mIoU"])), float(np.mean(db["primary"]["mIoU"])),
               float(np.mean(db["fused"]["fragments_mean"])))
    if ca:
        matrix_row("C′⊘A", ca["primary"]["fragments_mean"], ca["fused"]["mIoU"],
                   ca["primary"]["mIoU"], ca["fused"]["fragments_mean"])
    if da:
        matrix_row("D′⊘A", da["primary"]["fragments_mean"], da["fused"]["mIoU"],
                   da["primary"]["mIoU"], da["fused"]["fragments_mean"])
    if not (ca and da):
        L.append("| C′⊘A, D′⊘A | *(veto-A lean run pending on Tower)* | — | — |")
    L.append("")
    if ca and da:
        cb_fr = 100*(np.mean(cb["fused"]["fragments_mean"])-np.mean(cb["primary"]["fragments_mean"]))/np.mean(cb["primary"]["fragments_mean"])
        ca_fr = 100*(ca["fused"]["fragments_mean"]-ca["primary"]["fragments_mean"])/ca["primary"]["fragments_mean"]
        L.append(f"Veto **B (=CE+Dice baseline) prunes more than A (=CE)** (C′⊘B {cb_fr:.1f} % vs "
                 f"C′⊘A {ca_fr:.1f} %) → the BRATS-faithful veto = baseline B, hence canonical = C′⊘B.")
        L.append("")

    # ---- §4/§5 full veto-B characterisations ----
    def full_section(num, tag, data, bootj, primlabel):
        L.append(f"## {num}. {tag} — FULL (fused vs primary {primlabel}, paired n=3)")
        L.append("| metric | Δ | p (seed paired-t) | image-bootstrap |")
        L.append("|---|---|---|---|")
        bp = ""
        if bootj:
            k = list(bootj["pairwise"])[0]
            r = bootj["pairwise"][k]
            # bootstrap delta is primary−fused; flip sign to fused−primary
            bp = f"Δ {(-r['delta_miou'])*100:+.2f}, p = {r['p_two_sided']:.3f} → " + \
                 ("**neutral**" if r["p_two_sided"] >= 0.05 else "significant")
        for m, unit, scale in [("mIoU", " pp", 100), ("fragments_mean", " %", None),
                               ("boundary_f1_mean", " pp", 100), ("trimap_mIoU_mean", " pp", 100)]:
            prim, fused = data["primary"][m], data["fused"][m]
            t, p = stats.ttest_rel(fused, prim)
            star = " *" if p < 0.05 else ""
            if m == "fragments_mean":
                dpct = 100*(fused.mean()-prim.mean())/prim.mean()
                dabs = fused.mean()-prim.mean()
                dcell = f"{dpct:+.1f} % ({dabs:+.0f}/img)"
                bcell = "—"
            else:
                dcell = f"{(fused.mean()-prim.mean())*scale:+.3f}{unit}"
                bcell = bp if m == "mIoU" else "negligible" if abs((fused.mean()-prim.mean())*100) < 0.3 else "notable"
            mlab = {"mIoU": "mIoU", "fragments_mean": "fragments",
                    "boundary_f1_mean": "Boundary F1", "trimap_mIoU_mean": "Trimap IoU"}[m]
            L.append(f"| {mlab} | {dcell} | {fmt_p(p)}{star} | {bcell} |")
        L.append("")

    full_section(4, "Canonical consensus C′⊘B", cb, bcb, "C′")
    L.append("→ **Pure spatial-coherence cleanup**: removes ≈18 % of connected-component fragments "
             "at zero mIoU cost and boundary-quality-neutral — the tight 2-D analogue of the BRATS "
             "DistMap⊘Baseline result (region-overlap-neutral; the gain is on fragmentation, which "
             "mIoU is blind to).")
    L.append("")
    full_section(5, "Contrast consensus D′⊘B", db, bdb, "D′")
    L.append("→ Same fragment prune + zero mIoU cost, BUT a **Dice-character shift** (Boundary F1 ↓, "
             "Trimap ↑): reassigning D′'s (no-Dice) fragments to B's Dice-trained labels Dice-ifies "
             "the output. C′⊘B (both with Dice) has no such side-effect → cleaner canonical choice.")
    L.append("")

    # ---- takeaways ----
    L.append("## Takeaways to encode in the manuscript")
    L.append("1. Baseline = **B (CE+Dice)** (= nnU-Net/BRATS default); A (CE) and D′ (CE+DistMap) are "
             "the Cityscape Dice-axis ablations.")
    L.append("2. §5.1: report BOTH tests (image-bootstrap primary + seed robustness).")
    L.append("3. §5.5: **canonical consensus = C′⊘B** (pure fragment cleanup, mIoU- and "
             "boundary-neutral) — the tight BRATS analogue; D′⊘B shown as the contrast + Dice shift.")
    L.append("4. §4.2: pre-specify the primary endpoint (mIoU) + the paired image-bootstrap protocol.")
    L.append("5. Cross-paper thesis: a boundary-aware model (DistMap, like Kervadec in paper2) + a "
             "consensus veto removes fragmentation at no mIoU cost — holds in 2-D (Cityscapes) and "
             "3-D (BRATS).")
    L.append("")

    OUT.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

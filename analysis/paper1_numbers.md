# paper1 v0.1.0 — consolidated numbers (source of truth for the manuscript)

Val 500 img, 3 seeds (42/123/456), epoch 160. mIoU = official cityscapesscripts (bit-exact). Variants: A=CE, B=CE+Dice (=nnU-Net/BRATS baseline), C′=CE+Dice+DistMap, D′=CE+DistMap; DistMap = signed-distance-transform auxiliary regression head. A/B shared with paper2 (Kervadec). Raw JSONs in results/ and analysis/.

## 1. Variant singletons (mean over 3 seeds)
| | loss | mIoU | Boundary F1 | Trimap IoU | fragments/img |
|---|---|---|---|---|---|
| A | CE | 81.28 | **77.24** | 52.17 | 734.4 |
| B | CE+Dice (= nnU-Net/BRATS baseline) | 81.09 | 76.39 | 53.82 | 615.3 |
| C′ | CE+Dice+DistMap (SDT) | 80.89 | 76.43 | 53.84 | 630.5 |
| D′ | CE+DistMap (SDT) | **81.64** | 77.17 | 52.32 | 759.5 |

Dice axis: with-Dice (B,C′) ≈ 623 fragments/img vs no-Dice (A,D′) ≈ 747 — the DistMap term itself barely moves fragment count (C′−B = +15, D′−A = +25); like paper2, fragmentation is governed by the Dice axis, not the auxiliary boundary term.

## 1b. Convergence crossover (epoch 10 → 160)
Phase-1 small-ep checkpoints are **seed 42 only** (n=1, illustrative); epoch-160 is the 3-seed mean. Intermediate val curves are unavailable (checkpoints purged), so the crossover is shown as two points, not a continuous curve.

| variant | mIoU @ epoch 10 (seed 42) | mIoU @ epoch 160 (3-seed) |
|---|---|---|
| C′ (CE+Dice+DistMap) | **78.17** | 80.89 |
| D′ (CE+DistMap) | 75.59 | **81.64** |

At epoch 10 C′ leads D′ by +2.58 mIoU (Dice helps early); by epoch 160 D′ overtakes C′ by +0.75 — the same late-training crossover as paper2 (the no-Dice variant catches up and wins mIoU).

## 2. Variant pairwise mIoU — two tests (report BOTH, no cherry-pick)
**Primary = paired image-bootstrap (n=500, B=10 000, Holm)** — `analysis/paper1_bootstrap_variants.json`. **Secondary = seed paired-t (n=3, Holm)** (underpowered, robustness only).

| pair (X−Y) | Δ mIoU | bootstrap p (Holm) | seed-t p (Holm) |
|---|---|---|---|
| A−B | +0.19 | 0.325 (0.494) | 0.071 (0.284) |
| A−C′ | +0.39 | 0.044 (0.131) | 0.366 (0.731) |
| A−D′ | -0.36 | 0.014 (0.055) | 0.026 (0.156) |
| B−C′ | +0.20 | 0.247 (0.494) | 0.652 (0.731) |
| B−D′ | -0.55 | 0.009 (0.046) * | 0.038 (0.189) |
| C′−D′ | -0.75 | <0.001 (0.001) * | 0.115 (0.344) |

D′ (CE+DistMap, no Dice) has the highest mIoU and significantly beats B and C′ on the primary test — the same Dice-axis direction as paper2 (the no-Dice variant wins mIoU while trading boundary quality, see §4–5).

## 3. Consensus ablation matrix (mIoU + fragments)
| pairing | mIoU | ΔmIoU | Δfragments |
|---|---|---|---|
| **C′⊘B (canonical)** | 81.01 | +0.124 pp | -18.1 % |
| D′⊘B (contrast) | 81.65 | +0.005 pp | -18.7 % |
| C′⊘A | 81.15 | +0.262 pp | -7.7 % |
| D′⊘A | 81.74 | +0.097 pp | -13.8 % |

Veto **B (=CE+Dice baseline) prunes more than A (=CE)** (C′⊘B -18.1 % vs C′⊘A -7.7 %) → the BRATS-faithful veto = baseline B, hence canonical = C′⊘B.

## 4. Canonical consensus C′⊘B — FULL (fused vs primary C′, paired n=3)
| metric | Δ | p (seed paired-t) | image-bootstrap |
|---|---|---|---|
| mIoU | +0.124 pp | 0.160 | Δ +0.12, p = 0.093 → **neutral** |
| fragments | -18.1 % (-114/img) | 0.019 * | — |
| Boundary F1 | -0.049 pp | 0.407 | negligible |
| Trimap IoU | +0.077 pp | 0.138 | negligible |

→ **Pure spatial-coherence cleanup**: removes ≈18 % of connected-component fragments at zero mIoU cost and boundary-quality-neutral — the tight 2-D analogue of the BRATS DistMap⊘Baseline result (region-overlap-neutral; the gain is on fragmentation, which mIoU is blind to).

## 5. Contrast consensus D′⊘B — FULL (fused vs primary D′, paired n=3)
| metric | Δ | p (seed paired-t) | image-bootstrap |
|---|---|---|---|
| mIoU | +0.005 pp | 0.754 | Δ +0.00, p = 0.969 → **neutral** |
| fragments | -18.7 % (-142/img) | <0.001 * | — |
| Boundary F1 | -0.622 pp | 0.005 * | notable |
| Trimap IoU | +0.931 pp | 0.002 * | notable |

→ Same fragment prune + zero mIoU cost, BUT a **Dice-character shift** (Boundary F1 ↓, Trimap ↑): reassigning D′'s (no-Dice) fragments to B's Dice-trained labels Dice-ifies the output. C′⊘B (both with Dice) has no such side-effect → cleaner canonical choice.

## Takeaways to encode in the manuscript
1. Baseline = **B (CE+Dice)** (= nnU-Net/BRATS default); A (CE) and D′ (CE+DistMap) are the Cityscape Dice-axis ablations.
2. §5.1: report BOTH tests (image-bootstrap primary + seed robustness).
3. §5.5: **canonical consensus = C′⊘B** (pure fragment cleanup, mIoU- and boundary-neutral) — the tight BRATS analogue; D′⊘B shown as the contrast + Dice shift.
4. §4.2: pre-specify the primary endpoint (mIoU) + the paired image-bootstrap protocol.
5. Cross-paper thesis: a boundary-aware model (DistMap, like Kervadec in paper2) + a consensus veto removes fragmentation at no mIoU cost — holds in 2-D (Cityscapes) and 3-D (BRATS).

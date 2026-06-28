# cityscape-distmap-aux-regression

**Distance-Map Auxiliary Regression for Full-Resolution Cityscapes Segmentation: When Dice Helps and When It Doesn't**

Code, configs, per-class metrics, figures, and paper source for a controlled 2×2 loss ablation on Cityscapes at native resolution (1024×2048) with ConvNeXt-V2-Base + UPerNet, isolating a **distance-map (signed-distance-transform) auxiliary-regression head** crossed with the Dice term.

[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21006235.svg)](https://doi.org/10.5281/zenodo.21006235)

> **TL;DR.** Add a per-class signed-distance-transform (SDT) auxiliary-regression head to a strong CE + Dice Cityscapes baseline and the converged-mIoU ranking flips between short and long training. At 160 epochs the **no-Dice variant D′ (CE + DistMap) has the highest mean mIoU (81.64 ± 0.27)** and significantly beats both the CE + Dice baseline and the joint variant on a paired image-bootstrap. The DistMap auxiliary, unlike a boundary *loss*, does **not** sharpen contours beyond plain CE — its gain is representation shaping, not edge sharpening. A connected-component consensus filter (canonical **C′⊘B**) prunes **−18.1 % of spurious fragments at no mIoU cost** and boundary-neutral.

This is the **DistMap arm** of a mirrored two-method study. The Kervadec boundary-*loss* sibling (same backbone, same protocol, same consensus filter) lives at [guillaume-cassez/city-scape](https://github.com/guillaume-cassez/city-scape).

---

## The four variants (clean 2×2)

The setup is narrow on purpose: take the canonical CE + Dice recipe used on Cityscapes and cross the **Dice axis** with the **DistMap axis** (a second auxiliary head that regresses, per class, the normalised SDT of the ground-truth mask by a masked MSE at a fixed weight λ = 1.0 — the 2-D transcription of the BRATS *Distance-Map Auxiliary Loss*). The head is dropped at inference (zero test-time cost). Four variants × 3 seeds × 160 epochs = twelve runs, evaluated at epoch 160.

| Variant | Loss | Dice | DistMap |
|---|---|:---:|:---:|
| **A**  | CE                       | —   | —   |
| **B**  | CE + Dice (= nnU-Net / BRATS baseline) | yes | —   |
| **C′** | CE + Dice + DistMap (SDT) | yes | yes |
| **D′** | CE + DistMap (SDT)        | —   | yes |

A and B are shared with the Kervadec sibling paper; C′ and D′ add the SDT-regression branch.

---

## Key results (Cityscapes val, 500 img, 3 seeds, epoch 160)

Official `cityscapesscripts` mIoU (estimator validated bit-for-bit); 95 % CIs use the Student-t factor (n = 3); per-class Boundary F1 / Trimap IoU computed per class.

| | loss | mIoU | Boundary F1 | Trimap IoU | fragments/img |
|---|---|---|---|---|---|
| A  | CE                  | 81.28 ± 0.53 | **77.24** | 52.17 | 734.4 |
| B  | CE + Dice           | 81.09 ± 0.74 | 76.39 | 53.82 | 615.3 |
| C′ | CE + Dice + DistMap | 80.89 ± 0.93 | 76.43 | 53.84 | 630.5 |
| D′ | CE + DistMap        | **81.64 ± 0.27** | 77.17 | 52.32 | 759.5 |

**Short vs long training (the headline).** At epoch 10 the joint variant C′ leads (78.17 mIoU, +2.26 over B), and D′ actually *trails* plain CE (75.59). By epoch 160 the picture flips: D′ (no Dice) reaches the highest mean mIoU and **overtakes C′ by +0.75**. A short ablation systematically picks the wrong variant on this task.

**mIoU significance — paired image-bootstrap (primary; n = 500, B = 10 000, Holm), 3-seed paired-t (robustness).** Under the image-bootstrap, **D′ beats the CE + Dice baseline B (Δ = +0.55, p = 0.046) and the joint variant C′ (Δ = +0.75, p = 0.001)**; D′ > A is directionally consistent across all three seeds but borderline (Δ = +0.36, Holm p = 0.055).

**Contour metrics split along the Dice axis.** The non-Dice variants (A, D′) lead **Boundary F1** (A − B = +0.85, A − C′ = +0.81; Holm p ≤ 0.008); the Dice variants (B, C′) lead **Trimap IoU** (B − D′ = +1.50, C′ − D′ = +1.52; Holm p ≤ 0.012). Dice trades edge sharpness for near-boundary region coherence.

**Honest negative result.** Unlike the Kervadec boundary *loss* (the sibling study), the DistMap auxiliary does **not** sharpen contours beyond plain CE: A and D′ are tied on Boundary F1 (Δ = +0.07, n.s.). The DistMap converged-mIoU gain comes from representation shaping, not contour sharpening.

**Per-class breakdown.** D′ dominates large-extent structured classes (truck +5.54, bus +2.59, motorcycle +1.73, wall +1.31 mIoU vs B) where the SDT field gives a consistent gradient; B preserves thin signal-rich classes (traffic light +1.83, train +0.85, traffic sign +0.82 mIoU vs D′) where Dice anchors small footprints against CE class imbalance.

---

## The consensus filter (canonical C′⊘B)

A connected-component **consensus filter** adapted from our BRATS work: per class, any connected component of a "generalist" segmentation with no same-class overlap in a "veto" variant is reassigned to the veto's label, pruning hallucinated fragments. Cityscapes-specific differences (2D connectivity, 19 flat classes, reassignment instead of zeroing, thin-class protection) are documented in `src/postprocessing/consensus.py` and §5.5 of the paper.

The **canonical pairing is C′⊘B** — the DistMap-with-Dice generalist (C′ = the literal 2-D transcription of the BRATS DistMap model) vetoed by the CE + Dice baseline (B = nnU-Net default) — the most faithful 2-D transcription of the BRATS DistMap⊘Baseline rule in the series. Full ablation matrix (mean over 3 seeds):

| pairing | mIoU | Δ mIoU vs primary | Δ fragments |
|---|---|---|---|
| **C′⊘B (canonical)** | 81.01 | +0.124 pp | **−18.1 %** |
| D′⊘B (contrast) | 81.65 | +0.005 pp | −18.7 % |
| C′⊘A | 81.15 | +0.262 pp | −7.7 % |
| D′⊘A | 81.74 | +0.097 pp | −13.8 % |

Every pairing is **mIoU-neutral** (|Δ mIoU| ≤ 0.27 pp); the baseline veto B prunes ≈ 2× more fragments than the CE-only veto A. **C′⊘B** removes −18.1 % of fragments (−114/img) at **zero mIoU cost** (seed paired-t p = 0.160; image-bootstrap Δ = +0.12, p = 0.093) and **no meaningful boundary-quality cost** (Boundary F1 −0.049 pp, Trimap IoU +0.077 pp, both < 0.1 pp) — a **pure spatial-coherence cleanup**, the tight 2-D analogue of the BRATS DistMap⊘Baseline result. The contrast pairing D′⊘B gives the same prune and mIoU-neutrality but carries a Dice-axis shift (Boundary F1 −0.622 pp, Trimap +0.931 pp), so C′⊘B is the *pure* consensus.

`scripts/evaluate_consensus_matrix.py` / `scripts/evaluate_consensus.py` reproduce these from the trained checkpoints; the filter is unit-tested (19/19).

---

## Repository layout

```
├── paper.md                Paper source (English, Markdown)
├── paper.pdf               Compiled paper EN (Pandoc + XeLaTeX)
├── paper_fr.md             Paper source (French, Markdown)
├── paper_fr.pdf            Compiled paper FR (Pandoc + XeLaTeX)
├── header.tex              LaTeX header used by build.sh
├── build.sh                Pandoc + XeLaTeX build (paper.md / paper_fr.md)
├── README.md               This file
├── CHANGELOG.md            Version history
├── LICENSE                 MIT
├── CITATION.cff            Machine-readable citation
├── src/                    Self-contained method + metric + consensus code
│   ├── losses/distmap_aux.py            DistMap SDT auxiliary-regression head + masked-MSE loss
│   ├── models/builder.py                Backbone + UPerNet (+ DistMap head) builder
│   ├── metrics/segmentation_metrics.py  mIoU + per-class Boundary F1 / Trimap IoU
│   └── postprocessing/consensus.py      CC consensus filter (BRATS-adapted)
├── tests/
│   ├── test_official_miou.py       mIoU == official cityscapesscripts (bit-for-bit)
│   └── test_metrics_consensus.py   Synthetic unit tests for the metric / consensus code
├── scripts/                Re-eval, bootstrap, consensus, table/figure pipeline
│   ├── reeval_official.py           Re-eval: official cityscapesscripts mIoU + per-class contour
│   ├── aggregate_official.py        Tables (Student-t CI + Holm tests) from the re-eval
│   ├── bootstrap_miou.py            Paired image-bootstrap mIoU (primary significance test)
│   ├── dump_perimage_cm.py          Per-image confusion matrices (bootstrap input)
│   ├── evaluate_consensus.py        Variant-pair CC veto evaluation
│   ├── evaluate_consensus_matrix.py Full four-pairing consensus ablation matrix
│   ├── paper1_assemble_numbers.py   Consolidate every reported number
│   ├── paper1_tables.py             Build the paper tables
│   ├── paper1_figures.py            Build the paper figures
│   ├── paper1_analysis.sh           End-to-end analysis driver
│   └── paper1_vetoA.sh              CE-only (A) veto ablation driver
├── analysis/               Bootstrap + significance JSON + numbers source of truth
│   ├── paper1_numbers.md            Consolidated numbers (source of truth)
│   ├── paper1_bootstrap_variants.json   Paired image-bootstrap, all variant pairs
│   ├── paper1_bootstrap_CpvetoB.json    Image-bootstrap for C′⊘B consensus
│   └── paper1_bootstrap_DpvetoB.json    Image-bootstrap for D′⊘B consensus
├── data/                   Per-run CSV + consensus JSON exports (no raw images)
│   ├── pilot_results_official.csv   epoch-160 official mIoU + per-class contour
│   ├── paper1_consensus_CpB_full.json / paper1_consensus_DpB_full.json
│   └── paper1_consensus_CpA_lean.json / paper1_consensus_DpA_lean.json
└── figures/                Paper figures + table CSVs
    ├── fig_perclass.{png,pdf}       Per-class IoU at epoch 160, 4 variants
    ├── table_final.csv              Global metrics + Student-t 95 % CI at epoch 160
    ├── table_significance.csv       Holm-corrected paired tests (all pairs)
    ├── table_perclass.csv           Per-class IoU + std at epoch 160 (official)
    └── pilot_results_official.csv   epoch-160 official mIoU + per-class contour
```

Cityscapes raw images are **not** redistributed (Cityscapes Dataset Terms of Use). Obtain them from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/).

---

## Try the DistMap auxiliary on your own pipeline

The DistMap head regresses the per-class **signed distance transform** (negative inside, positive outside) of the ground-truth mask, normalised to [−1, 1], by a masked MSE. Pre-compute the per-class SDT once offline, then add a small head + one MSE term to your existing CE / CE+Dice pipeline:

```python
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

def signed_distance_transform(mask: np.ndarray, tau: float = 127.0) -> np.ndarray:
    """Per-class signed DT. mask shape (C, H, W) one-hot binary.
    Negative inside the region, positive outside; clipped to [-tau, tau]
    then normalised to [-1, 1] per class (the DistMap regression target).
    """
    sdt = np.empty(mask.shape, dtype=np.float32)
    for c in range(mask.shape[0]):
        m = mask[c].astype(bool)
        if m.any() and not m.all():
            inside  = distance_transform_edt(m)
            outside = distance_transform_edt(~m)
            sdt[c]  = np.clip(-inside + outside, -tau, tau)
        else:
            sdt[c] = 0.0
    sdt /= tau  # -> [-1, 1]
    return sdt

def distmap_aux_loss(pred_sdt: torch.Tensor, target_sdt: torch.Tensor,
                     valid: torch.Tensor) -> torch.Tensor:
    """Masked MSE between the predicted (tanh) SDT and the target SDT.
    pred_sdt, target_sdt : (B, C, H, W) in [-1, 1]
    valid                : (B, 1, H, W) {0,1} valid-pixel mask (0 on void/ignore)
    """
    se = (pred_sdt - target_sdt) ** 2
    return (se * valid).sum() / valid.sum().clamp_min(1.0)
```

Add a `Conv3×3 → BN → ReLU → Dropout(0.1) → Conv1×1 → tanh` head on the stride-16 features, upsample to full resolution, and combine as
`loss = ce(logits, target) + dice(logits, target) + 1.0 * distmap_aux_loss(pred_sdt, target_sdt, valid)` (variant C′; drop the `dice` term for D′). The head is removed at inference. See `src/losses/distmap_aux.py` for the released implementation.

---

## Reproduce the numbers

```bash
# 1. (Optional) Re-train any variant (requires Cityscapes fine annotations + ~28 h on a 96 GB GPU)
#    Hydra launch uses +experiment= :
#    python scripts/train.py --config-name=config +experiment=pilot_fullres_C_ce_dice_distmap \
#        training.epochs=160 "experiment.seeds=[42,123,456]"

# 2. Re-evaluate all twelve runs (official cityscapesscripts mIoU + per-class contour)
python scripts/reeval_official.py
python scripts/aggregate_official.py        # -> figures/table_final.csv, table_significance.csv, table_perclass.csv

# 3. Primary significance test (paired image-bootstrap over the 500 val images)
python scripts/dump_perimage_cm.py
python scripts/bootstrap_miou.py            # -> analysis/paper1_bootstrap_variants.json

# 4. Consensus filter (canonical C′⊘B + full four-pairing matrix)
python scripts/evaluate_consensus_matrix.py # -> data/paper1_consensus_*.json
python scripts/paper1_assemble_numbers.py   # -> analysis/paper1_numbers.md (source of truth)

# 5. Tables + figures
python scripts/paper1_tables.py
python scripts/paper1_figures.py            # -> figures/fig_perclass.{png,pdf}
```

`analysis/paper1_numbers.md` is the consolidated source of truth for every number reported in the paper.

---

## Cite

```bibtex
@misc{cassez2026distmap,
  title   = {Distance-Map Auxiliary Regression for Full-Resolution Cityscapes
             Segmentation: When Dice Helps and When It Doesn't},
  author  = {Cassez, Guillaume},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.21006235},
  note    = {Independent research}
}
```

**Concept DOI** (always resolves to the latest version): [10.5281/zenodo.21006235](https://doi.org/10.5281/zenodo.21006235). This **v0.1.0** release: [10.5281/zenodo.21006236](https://doi.org/10.5281/zenodo.21006236).

---

## About the author

I'm **Guillaume Cassez** ([ORCID 0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931)), and I built this project in 2026 as **independent research** (outside any institutional framework). It is part of a mirrored series on geometry-aware training signals for segmentation: the **DistMap auxiliary-regression arm** here, and the **Kervadec boundary-loss sibling** ([guillaume-cassez/city-scape](https://github.com/guillaume-cassez/city-scape)), under an identical backbone, protocol and consensus filter.

**I'm currently looking for opportunities**, ideally:

- **ML / research engineering** in computer vision, autonomous-driving perception, or applied research
- **R&D positions** (full-time, contract, industrial post-doc) where loss design, training dynamics, or controlled ablation are first-class concerns
- **MLOps / engineering** in vision-heavy products

→ [cassez.guillaume@gmail.com](mailto:cassez.guillaume@gmail.com)
→ [guillaume-cassez.fr](https://guillaume-cassez.fr)
→ [ORCID 0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931)

For technical discussion on this work, please open a GitHub issue — it helps future readers.

---

## License

The **code** in this repository is released under the [MIT License](LICENSE).
The **figures and text of the paper** (paper.md and any derivative figures) are released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/): redistribution and remix permitted with attribution.
Cityscapes raw imaging data is **not** redistributed and remains under the [Cityscapes Dataset Terms of Use](https://www.cityscapes-dataset.com/).

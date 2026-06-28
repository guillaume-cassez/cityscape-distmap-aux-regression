# Changelog

All notable changes to this record are documented here. This project follows
[Zenodo concept versioning](https://help.zenodo.org/docs/deposit/manage-versions/):
the concept DOI (minted on the first Zenodo release) always resolves to the
latest version; earlier version DOIs remain permanently citable.

## v0.1.0 — 2026-06-28

First public release of the **DistMap auxiliary-regression** arm of the mirrored
Cityscapes loss-design study (the Kervadec boundary-loss sibling is released
separately).

- **First public release**: paper (EN + FR), self-contained method / metric /
  consensus code, per-class metrics for all twelve runs, figures, and the full
  table / figure / bootstrap pipeline.
- **2×2 (Dice × DistMap) ablation, 12 runs** — four loss variants (A: CE,
  B: CE+Dice, C′: CE+Dice+DistMap, D′: CE+DistMap) × 3 seeds (42/123/456) ×
  160 epochs, trained at native **1024×2048** with ConvNeXt-V2-Base + UPerNet.
  The DistMap head regresses the per-class signed distance transform (SDT) of the
  ground-truth mask by a masked MSE at a fixed weight (λ = 1.0); it is dropped at
  inference (zero test-time cost).
- **Official mIoU, bit-exact.** The reported mIoU is validated bit-for-bit against
  the official `cityscapesscripts` evaluation (`tests/test_official_miou.py`).
- **Per-class contour metrics.** Boundary F1 and Trimap IoU are computed **per
  class** on binary masks (not road-vs-rest).
- **Statistics.** mIoU significance uses a **paired image-bootstrap** over the 500
  val images (n = 500, B = 10 000) as the pre-specified primary test, with the
  3-seed **paired-t** as a robustness layer; all reported pairs use **Holm**
  correction. 95 % CIs use the Student-t factor (n = 3).
- **Headline finding.** Short-vs-long-training mismatch: at epoch 10 the joint
  variant C′ leads (78.17 mIoU, +2.26 over B), but by epoch 160 the no-Dice
  variant **D′ has the highest mean mIoU (81.64 ± 0.27)** and significantly beats
  the CE+Dice baseline B (image-bootstrap Δ = +0.55, p = 0.046) and the joint
  variant C′ (Δ = +0.75, p = 0.001). Short ablations are systematically
  misleading on this task.
- **Honest negative result.** Unlike the Kervadec boundary *loss* (sibling study),
  the DistMap auxiliary does **not** sharpen contours beyond plain CE — A and D′
  are tied on Boundary F1 (Δ = +0.07, n.s.). The DistMap converged-mIoU gain comes
  from representation shaping, not contour sharpening.
- **Canonical consensus C′⊘B.** A connected-component consensus filter (variant-pair
  veto, adapted from BRATS): the DistMap-with-Dice generalist C′ vetoed by the
  CE+Dice baseline B prunes **−18.1 % of spurious fragments at no mIoU cost**
  (seed paired-t p = 0.160; image-bootstrap p = 0.093) and boundary-neutral
  (Boundary F1 −0.049 pp, Trimap IoU +0.077 pp, both < 0.1 pp) — a pure
  spatial-coherence cleanup, the tight 2-D analogue of the BRATS DistMap⊘Baseline
  result. The full four-pairing ablation matrix is reported; the contrast pairing
  D′⊘B gives the same prune but carries a Dice-axis shift. The filter is
  unit-tested (19/19).
- **Caveat retained.** The seed-level contour deltas are at n = 3 (underpowered);
  reported alongside the image-bootstrap as a robustness layer, not as the primary
  evidence.

#!/bin/bash
# Paper-1 ANALYSIS (GPU) — run AFTER the reeval Step-1 finishes (one GPU job at a time).
# Produces everything the paper1 consensus story needs, mirroring paper2 v0.2.2:
#   (1) per-image 19x19 confusion matrices for A,B,Cp(C'),Dp(D')  (+C,D harmless) — the input
#       for the paired image-bootstrap (SPEC primary significance test). All co-located in
#       results/perimage_cm/ so bootstrap_miou.py can pair fused-vs-primary and variant-vs-variant.
#   (2) canonical consensus Cp veto B  (C'⊘B = BRATS-faithful DistMap⊘Baseline) FULL
#       (mIoU + fragments + Boundary F1 + Trimap) + the fused per-image cm (--dump-cm).
#   (3) Dp veto B (D'⊘B, the 2D-finding contrast) FULL + fused cm.
# Sequential (full-res inference, never stack). Detach with setsid; poll results/paper1_analysis.marker.
set -u
cd /home/ser/Bureau/City_Scape || exit 1
export CITYSCAPES_ROOT=/home/ser/datasets/cityscapes
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CITYSCAPE_CKPT_ROOT=reeval_ckpt_layout   # has A/B/C/D + symlinked Cp/Dp in the dir layout
L=results/paper1_analysis.log
CM=results/perimage_cm
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === paper1 analysis START ===" >"$L"

echo "[$(ts)] (1) dump per-image cm (A,B,Cp,Dp,C,D) -> $CM" >>"$L"
taskset -c 0-15 python3 -u scripts/dump_perimage_cm.py \
  --checkpoints "reeval_ckpt_layout/*/epoch_160.pth" --out-dir "$CM" >>"$L" 2>&1
echo "[$(ts)] dump-cm exit=$?" >>"$L"

echo "[$(ts)] (2) consensus Cp veto B (C'⊘B) FULL + dump-cm" >>"$L"
taskset -c 0-15 python3 -u scripts/evaluate_consensus.py --mode pair --primary Cp --veto B \
  --out results/paper1_consensus_CpB_full.json --dump-cm "$CM" >>"$L" 2>&1
echo "[$(ts)] CpB exit=$?" >>"$L"

echo "[$(ts)] (3) consensus Dp veto B (D'⊘B) FULL + dump-cm" >>"$L"
taskset -c 0-15 python3 -u scripts/evaluate_consensus.py --mode pair --primary Dp --veto B \
  --out results/paper1_consensus_DpB_full.json --dump-cm "$CM" >>"$L" 2>&1
echo "[$(ts)] DpB exit=$?" >>"$L"

echo "[$(ts)] === PAPER1 ANALYSIS DONE ===" >>"$L"
echo DONE > results/paper1_analysis.marker

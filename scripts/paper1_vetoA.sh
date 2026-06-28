#!/bin/bash
# Paper-1 consensus matrix completion: the veto-A pairings (C'⊘A, D'⊘A), LEAN (mIoU + fragments).
# Mirrors paper2 §3 (4-pairing matrix C/D ⊘ {A,B}). The point: veto B (=CE+Dice baseline) prunes
# ~2x more fragments than veto A (=CE) -> confirms the BRATS-faithful veto = baseline B, i.e. why
# the canonical paper1 consensus is C'⊘B. Lean = mIoU + fragments only (~10x faster than the full run).
# Sequential (one full-res GPU job at a time — never stack). Detach with setsid; poll results/paper1_vetoA.marker.
set -u
cd /home/ser/Bureau/City_Scape || exit 1
export CITYSCAPES_ROOT=/home/ser/datasets/cityscapes
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CITYSCAPE_CKPT_ROOT=reeval_ckpt_layout
L=results/paper1_vetoA.log
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === paper1 veto-A matrix START (lean) ===" >"$L"

echo "[$(ts)] (1) Cp veto A (C'⊘A) lean" >>"$L"
taskset -c 0-15 python3 -u scripts/evaluate_consensus.py --mode pair --primary Cp --veto A --lean \
  --out results/paper1_consensus_CpA_lean.json >>"$L" 2>&1
echo "[$(ts)] CpA exit=$?" >>"$L"

echo "[$(ts)] (2) Dp veto A (D'⊘A) lean" >>"$L"
taskset -c 0-15 python3 -u scripts/evaluate_consensus.py --mode pair --primary Dp --veto A --lean \
  --out results/paper1_consensus_DpA_lean.json >>"$L" 2>&1
echo "[$(ts)] DpA exit=$?" >>"$L"

echo "[$(ts)] === PAPER1 VETO-A DONE ===" >>"$L"
echo DONE > results/paper1_vetoA.marker

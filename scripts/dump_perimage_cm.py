#!/usr/bin/env python3
"""Dump PER-IMAGE 19x19 confusion matrices per checkpoint — input for the paired image-bootstrap.

reeval_official.py keeps only the dataset-level confusion matrix; the paired image-bootstrap
(SPEC validation protocol, decided 2026-06-18) needs one cm per val image per checkpoint, in a
fixed shared image order so the bootstrap can pair the same image across variants/seeds.

For each checkpoint <run>__epoch_160.pth it writes:
    <out_dir>/<run>__epoch_160.cm.npy   int64 (N_val, 19, 19), cm[i] = cm[gt, pred] of image i
and once:
    <out_dir>/val_image_order.json      the ordered image keys (alignment check)

Self-check per checkpoint: sum_i cm[i] reproduces the official dataset mIoU bit-for-bit
(|official - project| printed; must be < 1e-6), and the image order matches the first checkpoint.

Run on the GPU host (Tower), corrected src/ synced, cityscapesscripts installed. Use GPU1 +
E-cores to avoid contending with a P-core job on GPU0:
    CITYSCAPES_ROOT=/home/ser/datasets/cityscapes CUDA_VISIBLE_DEVICES=1 \
    taskset -c 16-27 python scripts/dump_perimage_cm.py \
        --checkpoints '/home/ser/Bureau/City_Scape/reeval_ckpt_layout/*__epoch_160.pth' \
        --out-dir results/perimage_cm
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.models import build_model
from src.data import build_dataloaders
from reeval_official import official_from_trainid_cm, parse_variant_seed

IGNORE, NC = 255, 19


def per_image_cm(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-image 19x19 confusion cm[gt, pred] with the EXACT void-masking of SegmentationMetrics."""
    k = (gt != IGNORE) & (gt >= 0) & (gt < NC)
    idx = NC * gt[k].astype(np.int64) + pred[k].astype(np.int64)
    return np.bincount(idx, minlength=NC * NC).reshape(NC, NC).astype(np.int64)


def miou_from_cm(cm: np.ndarray) -> float:
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    valid = union > 0
    iou = np.divide(inter, union, out=np.zeros(NC), where=valid)
    return float(iou[valid].mean()) if valid.any() else 0.0


def dump_ckpt(ckpt_path: str, device: torch.device, out_dir: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    OmegaConf.set_struct(cfg, False)
    try:
        cfg.model.backbone.pretrained = "none"
    except Exception:
        pass

    set_seed(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    channels_last = bool(cfg.training.get("channels_last", True))
    model = build_model(cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = build_dataloaders(cfg)["val"]
    cms, keys = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(images)
                if isinstance(out, tuple):
                    out = out[0]
            pred = out.argmax(1).cpu().numpy()
            lab = labels.numpy()
            names = batch.get("name") if isinstance(batch, dict) else None
            for b in range(pred.shape[0]):
                cms.append(per_image_cm(pred[b], lab[b]))
                keys.append(str(names[b]) if names is not None else f"idx{len(keys)}")

    arr = np.stack(cms, 0)
    variant, seed = parse_variant_seed(str(ckpt_path))
    run = f"{variant}_seed{seed}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{run}__epoch_160.cm.npy", arr)

    agg = arr.sum(0)
    off_miou, _ = official_from_trainid_cm(agg)        # official routine (cityscapesscripts)
    diff = abs(off_miou - miou_from_cm(agg))           # project convention, must match bit-for-bit
    return run, arr.shape, off_miou, diff, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, help="glob to <run>__epoch_160.pth")
    ap.add_argument("--out-dir", default="results/perimage_cm")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(args.checkpoints))
    if not ckpts:
        sys.exit(f"no checkpoints match: {args.checkpoints}")
    print(f"device={device}  {len(ckpts)} checkpoints", flush=True)

    ref_keys = None
    n_bad = 0
    for i, c in enumerate(ckpts):
        t0 = time.time()
        run, shape, off_miou, diff, keys = dump_ckpt(c, device, args.out_dir)
        if ref_keys is None:
            ref_keys = keys
            (Path(args.out_dir) / "val_image_order.json").write_text(json.dumps(ref_keys))
        aligned = (keys == ref_keys)
        flag = "" if (aligned and diff < 1e-6) else "  <-- CHECK"
        if not (aligned and diff < 1e-6):
            n_bad += 1
        print(f"[{i+1}/{len(ckpts)}] {run} shape={shape} mIoU_off={off_miou*100:.2f} "
              f"|d|={diff:.2e} aligned={aligned} ({time.time()-t0:.0f}s){flag}", flush=True)
    print(f"done — {len(ckpts)} dumped, {n_bad} need checking", flush=True)


if __name__ == "__main__":
    main()

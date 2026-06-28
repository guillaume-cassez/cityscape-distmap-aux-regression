"""
Consensus post-processing for multi-prediction Cityscapes segmentation.

Adapts the BRATS CC-consensus (connected-component veto) filter to Cityscapes.
In BRATS, a "generalist" prediction (DistMap) is vetoed by a "specialist"
(Baseline): per class, every connected component of the generalist with no
same-class overlap in the veto is dropped to background. This removes spurious
hallucinated fragments and improves boundary quality (HD95) at no Dice cost.

BRATS -> Cityscapes differences (why this is not a copy-paste):
  1. Dimensionality: 2D 8-connectivity instead of 3D 26-connectivity.
  2. Classes: 19 flat classes, no nested WT/TC/ET hierarchy — the veto runs
     over 19 independent class masks.
  3. No background class: in BRATS a dropped component becomes 0 (background).
     Cityscapes labels every pixel, so dropping = REASSIGNING to the veto's
     label there, not zeroing (zeroing would mean "road").
  4. Thin structures: pole / traffic light / traffic sign / fence are
     legitimately small, fragmented components that a naive veto would erase.
     They are protected by default (`protect_classes`) and/or by a `max_drop_size`
     cap (only fragments up to N pixels can be dropped). This is the dominant
     Cityscapes-specific failure mode absent from BRATS (where tumour cores are
     blob-like). BRATS reported 38.7% of cases degraded by an over-aggressive
     veto; on Cityscapes that risk is concentrated on thin classes, hence the
     guards.

Additionally provides a multi-seed majority vote and an agreement (uncertainty)
map — a Cityscapes-natural ensemble that directly attacks the high inter-seed
variance of an n=3 study (no direct BRATS analogue, which used 2 architectures
rather than seeds).

All functions operate on integer label maps (H, W) with trainId values in
[0, num_classes) and `ignore_index` (255) for void. NumPy only; no new deps.
"""

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.ndimage import label as ndi_label, generate_binary_structure

# Cityscapes thin / structurally-fragmented trainIds that a naive CC veto would
# erase: fence(4), pole(5), traffic light(6), traffic sign(7). Protected by default.
THIN_CLASSES = (4, 5, 6, 7)


def _struct(connectivity: int) -> np.ndarray:
    """8-connectivity (full 3x3) or 4-connectivity (plus-shaped) structuring element."""
    if connectivity == 8:
        return np.ones((3, 3), dtype=bool)
    return generate_binary_structure(2, 1)


def count_fragments(
    label_map: np.ndarray,
    num_classes: int = 19,
    ignore_index: int = 255,
    connectivity: int = 8,
) -> Dict[int, int]:
    """Number of connected components per class (BRATS-style fragment count).

    A spatial-coherence proxy independent of mIoU: fewer spurious fragments means
    a cleaner, more deployable mask. Returns {class_id: n_components} for the
    classes present in the map. The consensus filter is expected to lower this
    count (it only removes components, never adds them).
    """
    struct = _struct(connectivity)
    out: Dict[int, int] = {}
    for c in range(num_classes):
        m = label_map == c
        if not m.any():
            continue
        _, n = ndi_label(m, structure=struct)
        out[c] = int(n)
    return out


def cc_veto(
    primary: np.ndarray,
    veto: np.ndarray,
    num_classes: int = 19,
    connectivity: int = 8,
    max_drop_size: Optional[int] = None,
    protect_classes: Sequence[int] = THIN_CLASSES,
    reassign: str = "veto",
    ignore_index: int = 255,
) -> np.ndarray:
    """Drop connected components of `primary` not corroborated by `veto`.

    BRATS-style per-class CC veto adapted to 2D Cityscapes. For each non-protected
    class c, every connected component of (primary == c) with NO overlap with
    (veto == c) is removed and the pixels are reassigned (see `reassign`).

    Args:
        primary: label map to be filtered (the "generalist", e.g. variant D).
        veto:    corroborating label map (the "specialist", e.g. variant B).
        max_drop_size: if set, only components of at most this many pixels may be
            dropped; larger uncorroborated components are kept (assumed to be real
            objects, not hallucinations). None = BRATS-strict (drop any size).
        protect_classes: class ids never filtered (thin classes by default).
        reassign: 'veto'   -> dropped pixels take the veto's label (default; sound
                              because the component has zero overlap with veto==c);
                  'ignore' -> dropped pixels become `ignore_index` (abstention,
                              useful to quantify how much area the filter removes).

    Returns:
        A filtered copy of `primary` (never mutates the input).
    """
    if reassign not in ("veto", "ignore"):
        raise ValueError(f"reassign must be 'veto' or 'ignore', got {reassign!r}")
    struct = _struct(connectivity)
    out = primary.copy()
    protect = set(protect_classes)

    for c in range(num_classes):
        if c in protect:
            continue
        pmask = primary == c
        if not pmask.any():
            continue
        vmask = veto == c
        labeled, n = ndi_label(pmask, structure=struct)
        for cid in range(1, n + 1):
            cc = labeled == cid
            if (cc & vmask).any():
                continue  # corroborated -> keep
            if max_drop_size is not None and int(cc.sum()) > max_drop_size:
                continue  # too large to be a hallucination -> keep
            out[cc] = ignore_index if reassign == "ignore" else veto[cc]
    return out


def majority_vote(
    preds: Sequence[np.ndarray],
    num_classes: int = 19,
    ignore_index: int = 255,
) -> np.ndarray:
    """Pixel-wise majority vote across N label maps (ensemble of seeds/models).

    Cityscapes-natural ensemble that directly reduces the inter-seed variance of
    an n=3 study. Ties are broken by lowest class id (argmax convention). A pixel
    where no voter emits a valid class stays void.
    """
    if len(preds) == 0:
        raise ValueError("majority_vote needs at least one prediction")
    stack = np.stack(preds, axis=0)
    votes = np.zeros((num_classes,) + stack.shape[1:], dtype=np.int16)
    for c in range(num_classes):
        votes[c] = (stack == c).sum(axis=0)
    winner = votes.argmax(axis=0).astype(stack.dtype)
    winner[votes.sum(axis=0) == 0] = ignore_index
    return winner


def agreement_map(
    preds: Sequence[np.ndarray],
    num_classes: int = 19,
) -> np.ndarray:
    """Per-pixel agreement = fraction of voters that match the majority label.

    1.0 = unanimous, 1/N = maximal disagreement. An uncertainty proxy: regions of
    high disagreement concentrate likely errors and are the natural place to focus
    error analysis or active-learning labelling. Returns a float32 map in (0, 1].
    """
    stack = np.stack(preds, axis=0)
    n = stack.shape[0]
    votes = np.zeros((num_classes,) + stack.shape[1:], dtype=np.int16)
    for c in range(num_classes):
        votes[c] = (stack == c).sum(axis=0)
    return votes.max(axis=0).astype(np.float32) / n

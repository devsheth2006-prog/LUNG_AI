"""Lung segmentation.

Two methods:
* `classical` — HU thresholding + connected components + morphology. Deterministic,
  needs no training, and is the documented *baseline* the U-Net is compared against.
  It is not a validated model: it can leak into the trachea/bowel gas and can exclude
  dense juxtapleural tumours (mitigated by closing + hole filling).
* `unet` — 2-D/3-D U-Net trained on LUNA16 lung masks (training/train_segmentation.py).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology


def classical_lung_mask(hu: np.ndarray, spacing=(1.0, 1.0, 1.0), is_hu: bool = True,
                        threshold_hu: float = -320.0) -> np.ndarray:
    """Return a boolean lung mask (z, y, x)."""
    vol = hu.astype(np.float32)
    if is_hu:
        air = vol < threshold_hu
    else:  # 8-bit demonstration image: Otsu threshold instead of HU
        air = vol < filters.threshold_otsu(vol)

    # 1) remove air connected to the image border (outside the body), slice by slice
    body_air = np.zeros_like(air)
    for z in range(air.shape[0]):
        lab = measure.label(air[z], connectivity=1)
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        body_air[z] = air[z] & ~np.isin(lab, border)

    # 2) keep the largest 3-D components (two lungs, or one if they touch)
    lab3 = measure.label(body_air, connectivity=1)
    if lab3.max() == 0:
        return np.zeros_like(air, dtype=bool)
    sizes = np.bincount(lab3.ravel())
    sizes[0] = 0
    voxel_ml = float(np.prod(spacing)) / 1000.0
    order = np.argsort(sizes)[::-1]
    keep = [order[0]]
    if len(order) > 1 and sizes[order[1]] > 0.2 * sizes[order[0]] and sizes[order[1]] * voxel_ml > 50:
        keep.append(order[1])
    mask = np.isin(lab3, keep)

    # 3) remove trachea/main bronchi approximately: tiny upper-slice components are dropped
    # 4) closing + hole filling to re-include vessels and juxtapleural nodules
    r = max(1, int(round(4.0 / max(spacing[1], 0.5))))
    selem = morphology.disk(r)
    out = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        if mask[z].any():
            sl = ndimage.binary_closing(mask[z], structure=selem, border_value=0)
            out[z] = ndimage.binary_fill_holes(sl)
    return out


def split_left_right(mask: np.ndarray) -> np.ndarray:
    """Label map: 1 = patient right lung, 2 = patient left lung (LPS: higher x = left).

    If the two lungs are separate components, each is labelled by which side of the midline its
    centre lies on. If they touch (one large component, possibly plus small fragments), the mask is
    split at the gap between the lungs: the x position with the fewest lung voxels in the central
    part of the chest. Small fragments are labelled by the side they lie on."""
    out = np.zeros(mask.shape, np.uint8)
    if not mask.any():
        return out
    lab = measure.label(mask, connectivity=1)
    props = sorted(measure.regionprops(lab), key=lambda p: p.area, reverse=True)
    major = [p for p in props if p.area >= 0.05 * props[0].area]
    xs_all = np.argwhere(mask)[:, 2]
    lo, hi = np.percentile(xs_all, 5), np.percentile(xs_all, 95)
    if len(major) >= 2 and (major[0].centroid[2] < (lo + hi) / 2) != (major[1].centroid[2] < (lo + hi) / 2):
        split = (major[0].centroid[2] + major[1].centroid[2]) / 2
        for p in props:
            out[lab == p.label] = 1 if p.centroid[2] < split else 2
        return out
    # lungs touch: find the gap (fewest lung voxels) in the central 40% of the lung field
    counts = mask.sum(axis=(0, 1)).astype(float)
    a, b = int(lo + 0.3 * (hi - lo)), int(lo + 0.7 * (hi - lo)) + 1
    split = a + int(np.argmin(counts[a:b])) if b > a else int(np.median(xs_all))
    out[mask] = 1
    out[:, :, split:][mask[:, :, split:]] = 2
    return out


def lung_volume_ml(mask: np.ndarray, spacing) -> float:
    return float(mask.sum() * np.prod(spacing) / 1000.0)

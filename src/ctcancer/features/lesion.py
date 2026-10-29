"""Lesion measurements and approximate anatomical location.

Location is reported as side (right/left lung) + vertical third (upper/middle/lower) +
central/peripheral. True lobar assignment needs fissure/lobe segmentation, which is not
trained here, so we never report a lobe name.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .radiomics import shape_features


def describe_location(centroid_zyx, lung_lr: np.ndarray, lung_mask: np.ndarray) -> dict:
    z, y, x = (int(round(v)) for v in centroid_zyx)
    z = int(np.clip(z, 0, lung_mask.shape[0] - 1))
    y = int(np.clip(y, 0, lung_mask.shape[1] - 1))
    x = int(np.clip(x, 0, lung_mask.shape[2] - 1))
    side_val = int(lung_lr[z, y, x]) if lung_lr is not None else 0
    if side_val == 0 and lung_lr is not None:  # juxtapleural: majority label in a local neighbourhood
        r = 15
        box = lung_lr[max(z - r, 0):z + r + 1, max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        vals = box[box > 0]
        side_val = int(np.bincount(vals).argmax()) if vals.size else 0
    side = {1: "right lung", 2: "left lung"}.get(side_val, "lung side undetermined")
    zs = np.where(lung_mask.any(axis=(1, 2)))[0]
    if len(zs):
        # LPS: z increases towards superior (head) in DICOM patient coordinates
        frac = (z - zs.min()) / max(zs.max() - zs.min(), 1)
        third = "lower" if frac < 1 / 3 else ("middle" if frac < 2 / 3 else "upper")
    else:
        third = "unknown"
    # central vs peripheral: distance of the lesion from the lung boundary relative to lung width
    sl = lung_mask[z]
    if sl.any():
        dist = ndimage.distance_transform_edt(sl)
        depth = dist[y, x] / max(dist.max(), 1)
        zone = "peripheral" if depth < 0.35 else "central"
    else:
        zone = "unknown"
    return {"side": side, "vertical_third": third, "zone": zone,
            "text": f"{side}, {third} third, {zone} (approximate; lobe not determined)"}


def measure_lesion(mask: np.ndarray, spacing, shape: dict | None = None) -> dict:
    """Clinical-style measurements; pass precomputed `shape` features to avoid recomputation."""
    if shape is None:
        idx = np.argwhere(mask)
        if len(idx):
            lo, hi = idx.min(0), idx.max(0) + 1
            shape = shape_features(mask[tuple(slice(a, b) for a, b in zip(lo, hi))], spacing)
        else:
            shape = shape_features(mask, spacing)
    sf = shape
    c = ndimage.center_of_mass(mask) if mask.any() else (np.nan,) * 3
    return {
        "volume_cm3": sf["volume_mm3"] / 1000.0 if np.isfinite(sf["volume_mm3"]) else None,
        "volume_mm3": sf["volume_mm3"],
        "max_axial_diameter_mm": sf["max_axial_diameter_mm"],
        "equiv_diameter_mm": sf["equiv_diameter_mm"],
        "max_3d_diameter_mm": sf["max_3d_diameter_mm"],
        "sphericity": sf["sphericity"],
        "elongation": sf["elongation"],
        "centroid_zyx_vox": tuple(float(v) for v in c),
        "centroid_mm": tuple(float(v * s) for v, s in zip(c, spacing)),
    }

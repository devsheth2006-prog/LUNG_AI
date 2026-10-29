"""Hand-crafted radiomic features (shape, first-order, GLCM texture).

Implemented with numpy/scikit-image so the pipeline runs everywhere. Feature definitions
follow the IBSI naming as closely as practical. If PyRadiomics is installed,
`extract_pyradiomics` gives the IBSI-compliant reference implementation; use one extractor
consistently between training and inference — values are not interchangeable.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats
from scipy.spatial import ConvexHull, QhullError
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

SHAPE_KEYS = ["volume_mm3", "surface_area_mm2", "sphericity", "compactness", "equiv_diameter_mm",
              "max_3d_diameter_mm", "max_axial_diameter_mm", "elongation", "flatness"]


def _max_pairwise_distance(pts: np.ndarray) -> float:
    """Exact maximum Euclidean distance (Feret diameter) via the convex hull."""
    if len(pts) < 2:
        return 0.0
    try:
        pts = pts[ConvexHull(pts).vertices] if len(pts) > 3 else pts
    except (QhullError, ValueError):  # degenerate (collinear/coplanar) point sets
        pass
    if len(pts) > 4000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 4000, False)]
    return float(np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1).max()))


def shape_features(mask: np.ndarray, spacing) -> dict[str, float]:
    spacing = np.asarray(spacing, float)
    n = int(mask.sum())
    out = {k: float("nan") for k in SHAPE_KEYS}
    if n == 0:
        return out
    vol = n * float(np.prod(spacing))
    out["volume_mm3"] = vol
    out["equiv_diameter_mm"] = (6 * vol / np.pi) ** (1 / 3)
    try:
        padded = np.pad(mask.astype(np.uint8), 1)
        verts, faces, _, _ = measure.marching_cubes(padded, 0.5, spacing=tuple(spacing))
        area = float(measure.mesh_surface_area(verts, faces))
        out["surface_area_mm2"] = area
        out["sphericity"] = float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area)
        out["compactness"] = float(vol / np.sqrt(np.pi * area ** 3))
        out["max_3d_diameter_mm"] = _max_pairwise_distance(verts)
    except (ValueError, RuntimeError):
        pass
    coords = np.argwhere(mask) * spacing
    if len(coords) >= 4:
        ev = np.sort(np.linalg.eigvalsh(np.cov(coords.T)))[::-1]
        ev = np.maximum(ev, 1e-9)
        out["elongation"] = float(np.sqrt(ev[1] / ev[0]))
        out["flatness"] = float(np.sqrt(ev[2] / ev[0]))
    # longest in-plane diameter on any axial slice (closest analogue of the clinical RECIST diameter)
    best = 0.0
    for z in np.unique(np.argwhere(mask)[:, 0]):
        sl = mask[z]
        if sl.sum() < 2:
            continue
        edge = sl & ~ndimage.binary_erosion(sl)          # boundary pixels suffice for the max distance
        best = max(best, _max_pairwise_distance(np.argwhere(edge) * spacing[1:]))
    out["max_axial_diameter_mm"] = best
    return out


def first_order_features(hu: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    vals = hu[mask].astype(np.float64)
    if vals.size == 0:
        return {}
    hist, _ = np.histogram(vals, bins=64)
    p = hist[hist > 0] / vals.size
    return {
        "fo_mean": float(vals.mean()), "fo_std": float(vals.std()), "fo_median": float(np.median(vals)),
        "fo_p10": float(np.percentile(vals, 10)), "fo_p90": float(np.percentile(vals, 90)),
        "fo_min": float(vals.min()), "fo_max": float(vals.max()),
        "fo_skewness": float(stats.skew(vals)) if vals.size > 2 else 0.0,
        "fo_kurtosis": float(stats.kurtosis(vals)) if vals.size > 3 else 0.0,
        "fo_entropy": float(-(p * np.log2(p)).sum()),
        "fo_energy": float(((vals + 1024) ** 2).sum() / vals.size),
        "fo_frac_solid": float((vals > -300).mean()),   # solid vs ground-glass component
    }


def glcm_features(hu: np.ndarray, mask: np.ndarray, levels: int = 32, hu_range=(-1000, 400)) -> dict[str, float]:
    """GLCM features aggregated over axial slices and 4 directions (distance 1)."""
    if mask.sum() < 8:
        return {}
    lo, hi = hu_range
    q = np.clip(((hu - lo) / (hi - lo) * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)
    props = {k: [] for k in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]}
    for z in np.unique(np.argwhere(mask)[:, 0]):
        m = mask[z]
        if m.sum() < 4:
            continue
        ys, xs = np.where(m)
        crop = q[z, ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        cm = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        crop[~cm] = 0  # level 0 reserved for background; dropped below
        crop = np.where(cm, np.maximum(crop, 1), 0).astype(np.uint8)
        g = graycomatrix(crop, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels, symmetric=True)
        g[0, :, :, :] = 0
        g[:, 0, :, :] = 0
        if g.sum() == 0:
            continue
        g = g.astype(np.float64) / g.sum(axis=(0, 1), keepdims=True).clip(min=1)
        for k in props:
            vals = graycoprops(g, k)
            props[k].append(float(np.nanmean(vals)))
    return {f"glcm_{k}": float(np.nanmean(v)) if v else float("nan") for k, v in props.items()}


def extract_radiomics(hu: np.ndarray, mask: np.ndarray, spacing, shape: dict | None = None) -> dict[str, float]:
    """All hand-crafted features for one lesion. `hu` must be in Hounsfield Units.
    Computation is restricted to the lesion bounding box (+2 voxels) for speed."""
    if mask.any():
        idx = np.argwhere(mask)
        lo, hi = np.maximum(idx.min(0) - 2, 0), np.minimum(idx.max(0) + 3, mask.shape)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        hu, mask = hu[box], mask[box]
    feats = {}
    feats.update(shape if shape is not None else shape_features(mask, spacing))
    feats.update(first_order_features(hu, mask))
    feats.update(glcm_features(hu, mask))
    return feats


def extract_pyradiomics(hu: np.ndarray, mask: np.ndarray, spacing, params: dict | None = None) -> dict[str, float]:
    """Optional IBSI-compliant extraction with PyRadiomics (pip install pyradiomics)."""
    import SimpleITK as sitk
    from radiomics import featureextractor

    img = sitk.GetImageFromArray(hu.astype(np.float32))
    msk = sitk.GetImageFromArray(mask.astype(np.uint8))
    for im in (img, msk):
        im.SetSpacing(tuple(float(s) for s in spacing[::-1]))
    ext = featureextractor.RadiomicsFeatureExtractor(**(params or {"binWidth": 25}))
    res = ext.execute(img, msk)
    return {k: float(v) for k, v in res.items() if not k.startswith("diagnostics")}

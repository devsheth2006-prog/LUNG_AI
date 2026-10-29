"""Classical nodule-candidate generation (stage 1 of a two-stage detector).

Stage 1 (this file, no learning): dense structures inside the lungs, thin vessels removed by
morphological opening, filtered by size and shape. This is high-sensitivity / low-precision,
like the candidate generators used in LUNA16.

Stage 2 (training/train_classifier.py --task detection): a CNN scores each candidate as
nodule vs non-nodule using LUNA16 `candidates_V2.csv` labels. **Only stage-2 outputs are
probabilities.** Without a trained stage-2 model the app shows candidates with a heuristic
`blobness` index that is explicitly *not* a probability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage
from skimage import measure, morphology


@dataclass
class Candidate:
    id: int
    centroid_zyx: tuple[float, float, float]        # voxel coords in preprocessed space
    bbox_zyx: tuple[int, int, int, int, int, int]   # z0,y0,x0,z1,y1,x1
    equiv_diameter_mm: float
    volume_mm3: float
    elongation: float
    blobness: float                                  # heuristic 0..1, NOT a probability
    mask: np.ndarray | None = field(default=None, repr=False)  # bool mask cropped to bbox

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mask")
        return d


def generate_candidates(hu: np.ndarray, lung_mask: np.ndarray, spacing, *, min_d_mm: float = 3.0,
                        max_d_mm: float = 40.0, max_candidates: int = 30, density_hu: float = -400.0,
                        is_hu: bool = True) -> list[Candidate]:
    spacing = np.asarray(spacing, float)
    # include juxtapleural lesions: dilate lung mask a little
    search = ndimage.binary_dilation(lung_mask, iterations=2)
    if is_hu:
        dense = (hu > density_hu) & search
    else:
        thr = np.percentile(hu[lung_mask], 90) if lung_mask.any() else hu.mean()
        dense = (hu > thr) & search
    # opening with a ~3 mm ball removes vessels thinner than ~3 mm
    rad_vox = np.maximum(np.round(1.5 / spacing).astype(int), 1)
    ball = _ellipsoid(rad_vox)
    opened = ndimage.binary_opening(dense, structure=ball)
    lab = measure.label(opened, connectivity=1)
    voxel_mm3 = float(np.prod(spacing))
    cands: list[Candidate] = []
    for rp in measure.regionprops(lab):
        vol = rp.area * voxel_mm3
        d = (6 * vol / np.pi) ** (1 / 3)
        if not (min_d_mm <= d <= max_d_mm):
            continue
        coords_mm = rp.coords * spacing
        if len(coords_mm) >= 4:
            ev = np.sort(np.linalg.eigvalsh(np.cov(coords_mm.T)))[::-1]
            elong = float(np.sqrt(ev[0] / max(ev[-1], 1e-6)))
        else:
            elong = 1.0
        # touches lung border massively (chest wall)? fraction of voxels outside the lung mask
        sl = rp.slice
        local = lab[sl] == rp.label
        inside = lung_mask[sl][local].mean()
        if inside < 0.3:
            continue
        blob = float(np.clip(1.0 / elong, 0, 1) * min(1.0, inside + 0.3))
        # morphological reconstruction: the opening eroded the lesion border, so grow the
        # component back into the dense voxels it was carved from (one structuring element).
        pad = rad_vox + 1
        lo = np.maximum(np.array(rp.bbox[:3]) - pad, 0)
        hi = np.minimum(np.array(rp.bbox[3:]) + pad, lab.shape)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        local = ndimage.binary_dilation(lab[box] == rp.label, structure=ball) & dense[box]
        vol = float(local.sum() * voxel_mm3)
        d = (6 * vol / np.pi) ** (1 / 3)
        z0, y0, x0 = (int(v) for v in lo)
        z1, y1, x1 = (int(v) for v in hi)
        cands.append(Candidate(0, tuple(float(c) for c in rp.centroid), (z0, y0, x0, z1, y1, x1),
                               round(float(d), 1), round(vol, 1), round(elong, 2), round(blob, 3), local))
    cands.sort(key=lambda c: (c.blobness, c.volume_mm3), reverse=True)
    cands = cands[:max_candidates]
    for i, c in enumerate(cands):
        c.id = i + 1
    return cands


def _ellipsoid(r) -> np.ndarray:
    rz, ry, rx = (int(v) for v in r)
    z, y, x = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    return (z / max(rz, 1)) ** 2 + (y / max(ry, 1)) ** 2 + (x / max(rx, 1)) ** 2 <= 1.0


def full_mask(c: Candidate, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    z0, y0, x0, z1, y1, x1 = c.bbox_zyx
    if c.mask is not None:
        m[z0:z1, y0:y1, x0:x1] = c.mask
    return m


def nms_by_distance(cands: list[Candidate], spacing, min_dist_mm: float) -> list[Candidate]:
    kept: list[Candidate] = []
    sp = np.asarray(spacing)
    for c in cands:
        if all(np.linalg.norm((np.array(c.centroid_zyx) - np.array(k.centroid_zyx)) * sp) >= min_dist_mm for k in kept):
            kept.append(c)
    return kept

"""Geometric chest *software-test phantom*.

This is NOT patient data and NOT a realistic CT: it is a deterministic arrangement of
ellipsoids with textbook HU values (air −1000, lung −850, soft tissue 40, bone 700, one
solid sphere) used to unit-test the software path (loading, resampling, lung masking,
candidate detection, measurement accuracy, report rendering). Any model output on it is
meaningless and the app labels it accordingly.
"""
from __future__ import annotations

import numpy as np

from .io.volume import CTVolume


def make_phantom(shape=(80, 192, 192), spacing=(2.5, 1.5, 1.5), nodule_mm: float = 14.0,
                 nodule_hu: float = 20.0, seed: int = 0) -> tuple[CTVolume, np.ndarray]:
    """Return (volume, nodule_mask). Nodule in the patient-RIGHT lung (low x index in LPS)."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    sz, sy, sx = spacing
    z, y, x = np.meshgrid(np.arange(nz) * sz, np.arange(ny) * sy, np.arange(nx) * sx, indexing="ij")
    cz, cy, cx = nz * sz / 2, ny * sy / 2, nx * sx / 2
    vol = np.full(shape, -1000.0, np.float32)
    body = ((y - cy) / (ny * sy * 0.42)) ** 2 + ((x - cx) / (nx * sx * 0.46)) ** 2 <= 1
    vol[body] = 40.0
    for side in (-1, 1):
        lung = (((z - cz) / (nz * sz * 0.45)) ** 2 + ((y - cy * 0.95) / (ny * sy * 0.30)) ** 2 +
                ((x - (cx + side * nx * sx * 0.2)) / (nx * sx * 0.16)) ** 2) <= 1
        vol[lung & body] = -850.0
    spine = ((y - (cy + ny * sy * 0.3)) / 10) ** 2 + ((x - cx) / 10) ** 2 <= 1
    vol[spine & body] = 700.0
    # nodule: right lung (x below centre), mid-upper height
    nc = (cz + nz * sz * 0.1, cy * 0.95, cx - nx * sx * 0.2)
    nod = ((z - nc[0]) ** 2 + (y - nc[1]) ** 2 + (x - nc[2]) ** 2) <= (nodule_mm / 2) ** 2
    vol[nod] = nodule_hu
    vol += rng.normal(0, 12, shape).astype(np.float32)
    v = CTVolume(vol, spacing, (0.0, 0.0, 0.0), True, "array", "PHANTOM-TEST",
                 {"Modality": "CT", "phantom": True},
                 ["Synthetic software-test phantom — not patient data; model outputs on it are meaningless."])
    return v, nod

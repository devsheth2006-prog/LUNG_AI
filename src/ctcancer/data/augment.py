"""Anatomically plausible augmentation for lesion patches and slices.

Allowed: small rotations (±15° in-plane), mild isotropic scaling (±10 %), small translations
(±3 voxels — keeps the lesion centred), HU-scale intensity jitter (±3 %) and offset (±20 HU)
simulating scanner/kernel variation, low-amplitude Gaussian noise.
Flips: an axial *lesion patch* has no reliable laterality cue, so left-right and
anterior-posterior flips are allowed for patch classifiers. For whole-slice LUNG segmentation
left-right flips are allowed (masks flip consistently) but cranio-caudal flips and 90° rotations
are NOT, because they create anatomy that never occurs (heart on top, inverted diaphragm).
Elastic deformations are not used: they distort nodule shape, which is itself a malignancy cue.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class PatchAugment:
    def __init__(self, rot_deg: float = 15, scale: float = 0.10, shift: int = 3, hu_scale: float = 0.03,
                 hu_offset: float = 20, noise_hu: float = 10, flips: bool = True, seed: int | None = None):
        self.rot_deg, self.scale, self.shift = rot_deg, scale, shift
        self.hu_scale, self.hu_offset, self.noise_hu, self.flips = hu_scale, hu_offset, noise_hu, flips
        self.rng = np.random.default_rng(seed)

    def __call__(self, hu: np.ndarray, mask: np.ndarray | None = None):
        """hu: (z, y, x) HU patch. Geometric transforms applied identically to the mask."""
        r = self.rng
        ang = r.uniform(-self.rot_deg, self.rot_deg)
        hu = ndimage.rotate(hu, ang, axes=(1, 2), reshape=False, order=1, mode="nearest")
        if mask is not None:
            mask = ndimage.rotate(mask.astype(np.float32), ang, axes=(1, 2), reshape=False, order=0) > 0.5
        s = 1 + r.uniform(-self.scale, self.scale)
        hu = _zoom_keep(hu, s, 1)
        if mask is not None:
            mask = _zoom_keep(mask.astype(np.float32), s, 0) > 0.5
        sh = r.integers(-self.shift, self.shift + 1, size=3)
        hu = np.roll(hu, sh, axis=(0, 1, 2))
        if mask is not None:
            mask = np.roll(mask, sh, axis=(0, 1, 2))
        if self.flips:
            for ax in (1, 2):
                if r.random() < 0.5:
                    hu = np.flip(hu, ax)
                    mask = np.flip(mask, ax) if mask is not None else None
        hu = hu * (1 + r.uniform(-self.hu_scale, self.hu_scale)) + r.uniform(-self.hu_offset, self.hu_offset)
        hu = hu + r.normal(0, self.noise_hu, hu.shape)
        return np.ascontiguousarray(hu, dtype=np.float32), (np.ascontiguousarray(mask) if mask is not None else None)


def _zoom_keep(a: np.ndarray, s: float, order: int) -> np.ndarray:
    """Zoom about the centre and crop/pad back to the original shape."""
    z = ndimage.zoom(a, s, order=order, mode="nearest")
    out = np.full_like(a, a.min())
    sl_src, sl_dst = [], []
    for n_in, n_out in zip(a.shape, z.shape):
        if n_out >= n_in:
            o = (n_out - n_in) // 2
            sl_src.append(slice(o, o + n_in))
            sl_dst.append(slice(0, n_in))
        else:
            o = (n_in - n_out) // 2
            sl_src.append(slice(0, n_out))
            sl_dst.append(slice(o, o + n_out))
    out[tuple(sl_dst)] = z[tuple(sl_src)]
    return out


class SliceAugment:
    """For 2-D lung segmentation: small rotation/scale, left-right flip, intensity jitter."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        ang = self.rng.uniform(-10, 10)
        img = ndimage.rotate(img, ang, reshape=False, order=1, mode="nearest")
        mask = ndimage.rotate(mask.astype(np.float32), ang, reshape=False, order=0) > 0.5
        if self.rng.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        img = np.clip(img * self.rng.uniform(0.95, 1.05) + self.rng.normal(0, 0.01, img.shape), 0, 1)
        return np.ascontiguousarray(img, np.float32), np.ascontiguousarray(mask)

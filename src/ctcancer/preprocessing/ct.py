"""CT preprocessing: resampling, HU clipping, windowing, denoising, cropping and 2.5-D/3-D inputs.

Why not "just convert to RGB"?  A CT voxel is a calibrated physical measurement (HU) with a
range of ~4000 values. Saving to 8-bit RGB keeps only one window (~256 levels), discards the
calibration and makes thresholds like "air < -900 HU" meaningless. We keep float HU for all
quantitative steps and only map to [0, 1] with an explicit, documented window as the final
network input. When an ImageNet-pretrained 2-D CNN needs 3 channels we fill them with
*three adjacent axial slices* (2.5-D input) instead of copying one gray slice three times, so
the extra channels carry real through-plane context.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..io.volume import CTVolume


def clip_hu(arr: np.ndarray, lo: float = -1200, hi: float = 600) -> np.ndarray:
    return np.clip(arr, lo, hi)


def apply_window(arr: np.ndarray, level: float, width: float) -> np.ndarray:
    """Map HU to [0, 1] with a radiological window (level/width)."""
    lo, hi = level - width / 2, level + width / 2
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def normalize_window(arr: np.ndarray, is_hu: bool, level: float = -600, width: float = 1500) -> np.ndarray:
    """Lung window for HU data; min-max for non-HU demonstration images."""
    if is_hu:
        return apply_window(arr, level, width)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def denoise(arr: np.ndarray, method: str = "none") -> np.ndarray:
    """Mild, edge-preserving-ish noise reduction. Off by default: aggressive smoothing removes
    small nodules and alters texture features."""
    if method == "median":
        return ndimage.median_filter(arr, size=(1, 3, 3))
    if method == "gaussian":
        return ndimage.gaussian_filter(arr, sigma=(0, 0.7, 0.7))
    return arr


def resample_volume(arr: np.ndarray, spacing, new_spacing, order: int = 1) -> tuple[np.ndarray, tuple]:
    """Resample to a target voxel spacing (z, y, x). Linear for images, nearest for masks."""
    spacing = np.asarray(spacing, float)
    new_spacing = np.asarray(new_spacing, float)
    zoom = spacing / new_spacing
    out = ndimage.zoom(arr, zoom, order=order, mode="nearest", grid_mode=False)
    real_spacing = tuple(spacing * np.array(arr.shape) / np.array(out.shape))
    return out.astype(arr.dtype), tuple(float(v) for v in real_spacing)


def resample_mask(mask: np.ndarray, spacing, new_spacing) -> np.ndarray:
    out, _ = resample_volume(mask.astype(np.uint8), spacing, new_spacing, order=0)
    return out.astype(bool)


def crop_to_mask(arr: np.ndarray, mask: np.ndarray, margin: int = 5):
    """Crop to the bounding box of a mask (+margin). Returns crop and the slices used."""
    if not mask.any():
        return arr, tuple(slice(0, s) for s in arr.shape)
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, arr.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return arr[sl], sl


def resize_slice(img: np.ndarray, size: int, order: int = 1) -> np.ndarray:
    zoom = (size / img.shape[0], size / img.shape[1])
    return ndimage.zoom(img, zoom, order=order)


def extract_patch(arr: np.ndarray, center_zyx, size, fill: float = -1024.0) -> np.ndarray:
    """Extract a cube (or (z,y,x) box) centred on a voxel, padding outside the volume."""
    size = np.broadcast_to(np.asarray(size, int), (3,))
    c = np.round(np.asarray(center_zyx)).astype(int)
    lo = c - size // 2
    out = np.full(tuple(size), fill, dtype=arr.dtype)
    src_lo = np.maximum(lo, 0)
    src_hi = np.minimum(lo + size, arr.shape)
    if np.any(src_hi <= src_lo):
        return out
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    out[tuple(slice(a, b) for a, b in zip(dst_lo, dst_hi))] = \
        arr[tuple(slice(a, b) for a, b in zip(src_lo, src_hi))]
    return out


def slice_to_25d(vol_norm: np.ndarray, z: int) -> np.ndarray:
    """Three adjacent axial slices (z-1, z, z+1) stacked as channels -> (3, H, W)."""
    zs = np.clip([z - 1, z, z + 1], 0, vol_norm.shape[0] - 1)
    return vol_norm[zs].astype(np.float32)


@dataclass
class PreprocessedCT:
    hu: np.ndarray                 # resampled HU volume (clipped)
    norm: np.ndarray               # lung-window normalised [0,1]
    spacing: tuple                 # resampled spacing (z,y,x)
    original_spacing: tuple
    original_shape: tuple
    is_hu: bool
    steps: list[str] = field(default_factory=list)


def preprocess_volume(vol: CTVolume, cfg_pre: dict) -> PreprocessedCT:
    """Full preprocessing used at inference time (and mirrored in training scripts)."""
    steps = []
    arr = vol.array.astype(np.float32)
    spacing = vol.spacing
    if vol.is_hu:
        lo, hi = cfg_pre.get("hu_clip", [-1200, 600])
        arr = clip_hu(arr, lo, hi)
        steps.append(f"HU clipped to [{lo}, {hi}]")
    method = cfg_pre.get("denoise", "none")
    if method != "none":
        arr = denoise(arr, method)
        steps.append(f"denoise={method}")
    target = cfg_pre.get("target_spacing_mm", [1.0, 1.0, 1.0])
    if vol.is_volumetric and vol.quantitative:
        arr, spacing = resample_volume(arr, vol.spacing, target, order=1)
        steps.append(f"resampled {tuple(round(float(s), 2) for s in vol.spacing)} -> "
                     f"{tuple(round(float(s), 2) for s in spacing)} mm")
    w = cfg_pre.get("lung_window", {"level": -600, "width": 1500})
    norm = normalize_window(arr, vol.is_hu, w["level"], w["width"])
    steps.append(f"lung window L={w['level']} W={w['width']} -> [0,1]" if vol.is_hu else "min-max normalised (non-HU)")
    return PreprocessedCT(arr, norm, tuple(spacing), vol.spacing, vol.shape, vol.is_hu, steps)

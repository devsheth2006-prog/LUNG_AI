"""Overlay rendering (numpy -> RGB uint8) shared by the app and the report."""
from __future__ import annotations

import io

import numpy as np
from scipy import ndimage

LUNG_RGB = np.array([64, 156, 255]) / 255.0
LESION_RGB = np.array([255, 196, 0]) / 255.0


def window_slice(hu2d: np.ndarray, level: float, width: float) -> np.ndarray:
    lo, hi = level - width / 2, level + width / 2
    return np.clip((hu2d - lo) / (hi - lo), 0, 1)


def overlay(gray01: np.ndarray, lung: np.ndarray | None = None, lesion: np.ndarray | None = None,
            lung_alpha: float = 0.22, contour_only_lesion: bool = True) -> np.ndarray:
    rgb = np.repeat(np.clip(gray01, 0, 1)[..., None], 3, -1).astype(np.float32)
    if lung is not None and lung.any():
        m = lung.astype(bool)
        rgb[m] = (1 - lung_alpha) * rgb[m] + lung_alpha * LUNG_RGB
    if lesion is not None and lesion.any():
        m = lesion.astype(bool)
        if contour_only_lesion:
            edge = m & ~ndimage.binary_erosion(m, iterations=1)
            edge = ndimage.binary_dilation(edge)
            rgb[edge] = LESION_RGB
        else:
            rgb[m] = 0.5 * rgb[m] + 0.5 * LESION_RGB
    return (rgb * 255).astype(np.uint8)


def draw_box(rgb: np.ndarray, y0: int, x0: int, y1: int, x1: int, color=(255, 196, 0), t: int = 1) -> np.ndarray:
    out = rgb.copy()
    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, out.shape[0] - 1), min(x1, out.shape[1] - 1)
    out[y0:y0 + t, x0:x1] = color
    out[y1 - t + 1:y1 + 1, x0:x1] = color
    out[y0:y1, x0:x0 + t] = color
    out[y0:y1, x1 - t + 1:x1 + 1] = color
    return out


def to_png_bytes(rgb: np.ndarray, scale: int = 1) -> bytes:
    import cv2

    if scale != 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb)
    return buf.tobytes()


def representative_slice(arrays, lesion_id: int | None) -> int:
    if lesion_id is not None and (arrays.lesion_labels == lesion_id).any():
        zs = np.where((arrays.lesion_labels == lesion_id).any(axis=(1, 2)))[0]
        areas = [(arrays.lesion_labels[z] == lesion_id).sum() for z in zs]
        return int(zs[int(np.argmax(areas))])
    lz = np.where((arrays.lung_lr > 0).any(axis=(1, 2)))[0]
    return int(lz[len(lz) // 2]) if len(lz) else arrays.norm.shape[0] // 2


def isotropic_display(img: np.ndarray, spacing_yx) -> np.ndarray:
    """Stretch so pixels are square on screen (coronal/sagittal views)."""
    sy, sx = spacing_yx
    if abs(sy - sx) < 1e-3:
        return img
    order = 0 if img.dtype in (np.uint8, bool) and img.ndim == 2 else 1
    zoom = (sy / min(sy, sx), sx / min(sy, sx)) + ((1,) if img.ndim == 3 else ())
    return ndimage.zoom(img, zoom, order=order)


def png_buffer(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    return buf.getvalue()

"""PyTorch datasets over the cached patch/slice files produced by training/prepare_data.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..preprocessing.ct import extract_patch, apply_window


def patch_to_input(hu: np.ndarray, mode: str, crop: int | None = None) -> np.ndarray:
    """HU patch (z,y,x) -> network input. '2.5d': (3,H,W) adjacent slices; '3d': (1,D,H,W)."""
    if crop:
        hu = extract_patch(hu, np.array(hu.shape) // 2, crop, fill=-1024.0)  # padding with air
    x = apply_window(hu.astype(np.float32), -600, 1500)
    if mode == "2.5d":
        z = x.shape[0] // 2
        return x[z - 1:z + 2]
    return x[None]


class LesionPatchDataset(Dataset):
    def __init__(self, table: pd.DataFrame, root: str | Path, label_col: str, mode: str = "2.5d",
                 augment=None, crop: int | None = None):
        self.t = table.reset_index(drop=True)
        self.root, self.label_col, self.mode, self.augment, self.crop = Path(root), label_col, mode, augment, crop

    def __len__(self) -> int:
        return len(self.t)

    def load(self, i: int):
        d = np.load(self.root / self.t.patch_path[i])
        return d["hu"].astype(np.float32), (d["mask"] if "mask" in d else None)

    def __getitem__(self, i: int):
        hu, mask = self.load(i)
        if self.augment is not None:
            hu, mask = self.augment(hu, mask)
        x = torch.from_numpy(patch_to_input(hu, self.mode, self.crop).copy())
        y = int(self.t[self.label_col][i])
        return x, y


def _fit(a: np.ndarray, size: int) -> np.ndarray:
    """Centre crop / zero-pad a 2-D array to size×size WITHOUT rescaling (keeps 1 mm pixels)."""
    out = np.zeros((size, size), a.dtype)
    h, w = a.shape
    sy, sx = max((h - size) // 2, 0), max((w - size) // 2, 0)
    dy, dx = max((size - h) // 2, 0), max((size - w) // 2, 0)
    ch, cw = min(h, size), min(w, size)
    out[dy:dy + ch, dx:dx + cw] = a[sy:sy + ch, sx:sx + cw]
    return out


class SegSliceDataset(Dataset):
    """2-D (image, mask) pairs; for tumour segmentation the central slices of lesion patches."""

    def __init__(self, table: pd.DataFrame, root: str | Path, kind: str = "lung", augment=None, n_slices: int = 5,
                 size: int = 96):
        self.root, self.kind, self.augment, self.size = Path(root), kind, augment, size
        self.items: list[tuple[str, int | None]] = []
        for p in table.path if kind == "lung" else table.patch_path:
            if kind == "lung":
                self.items.append((p, None))
            else:  # several axial slices through each tumour patch
                for dz in range(-(n_slices // 2), n_slices // 2 + 1):
                    self.items.append((p, dz))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        p, dz = self.items[i]
        d = np.load(self.root / p)
        if self.kind == "lung":
            img, mask = d["img"].astype(np.float32), d["mask"]
        else:
            hu, m = d["hu"].astype(np.float32), d["mask"]
            zc = int(np.argwhere(m).mean(0)[0]) if m.any() else hu.shape[0] // 2
            z = int(np.clip(zc + dz, 0, hu.shape[0] - 1))
            img, mask = _fit(apply_window(hu[z], -600, 1500), self.size), _fit(m[z], self.size)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return torch.from_numpy(img[None].copy()), torch.from_numpy(mask.astype(np.int64).copy())

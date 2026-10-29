"""LUNA16 helpers: lung-mask slices for segmentation and candidate patches for detection.

LUNA16 identifies scans by SeriesInstanceUID. A few LIDC patients have two scans; supply
`data/raw/LUNA16/seriesuid_to_patient.csv` (derivable from the LIDC-IDRI metadata) so that
splits are made per patient. Without it we fall back to per-series grouping and log a warning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..preprocessing.ct import apply_window, extract_patch, resample_volume, resize_slice

log = get_logger(__name__)


def read_mhd(path: str | Path):
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr, tuple(img.GetSpacing()[::-1]), tuple(img.GetOrigin()[::-1])


def world_to_voxel(xyz_world, origin_zyx, spacing_zyx) -> np.ndarray:
    zyx = np.asarray(xyz_world, float)[::-1]
    return (zyx - np.asarray(origin_zyx)) / np.asarray(spacing_zyx)


def patient_map(root: Path) -> dict[str, str]:
    f = root / "seriesuid_to_patient.csv"
    if f.exists():
        m = pd.read_csv(f)
        return dict(zip(m.seriesuid.astype(str), m.patient_id.astype(str)))
    log.warning("seriesuid_to_patient.csv not found: grouping by series (a few LIDC patients have 2 scans).")
    return {}


def build_lung_seg_slices(root: str | Path, out_dir: str | Path, size: int = 192, every_n: int = 4,
                          limit: int | None = None) -> pd.DataFrame:
    """Axial slices + LUNA16 lung masks (values 3/4 = right/left lung) for the 2-D U-Net."""
    root, out_dir = Path(root), Path(out_dir)
    (out_dir / "slices").mkdir(parents=True, exist_ok=True)
    pmap = patient_map(root)
    rows = []
    mhds = sorted(root.glob("subset*/*.mhd"))[: limit or None]
    for path in mhds:
        uid = path.stem
        mpath = root / "seg-lungs-LUNA16" / f"{uid}.mhd"
        if not mpath.exists():
            continue
        img, sp, _ = read_mhd(path)
        msk, _, _ = read_mhd(mpath)
        norm = apply_window(img, -600, 1500)
        for z in range(0, img.shape[0], every_n):
            m = msk[z] > 0
            np.savez_compressed(out_dir / "slices" / f"{uid}_{z}.npz",
                                img=resize_slice(norm[z], size).astype(np.float16),
                                mask=resize_slice(m.astype(np.float32), size, order=0) > 0.5)
            rows.append({"slice_id": f"{uid}_{z}", "patient_id": pmap.get(uid, uid), "series_uid": uid,
                         "has_lung": bool(m.any()), "path": f"slices/{uid}_{z}.npz"})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "lung_slices.csv", index=False)
    return df


def build_candidate_patches(root: str | Path, out_dir: str | Path, target_spacing=(1.0, 1.0, 1.0),
                            patch_vox: int = 48, neg_per_scan: int = 40, seed: int = 0,
                            limit: int | None = None) -> pd.DataFrame:
    """Patches around LUNA16 candidates (class 1 = nodule). Negatives are subsampled per scan;
    the imbalance that remains is handled by the loss (focal / class weights)."""
    root, out_dir = Path(root), Path(out_dir)
    (out_dir / "cand").mkdir(parents=True, exist_ok=True)
    cands = pd.read_csv(root / "candidates_V2.csv")
    pmap = patient_map(root)
    rng = np.random.default_rng(seed)
    rows = []
    mhds = {p.stem: p for p in root.glob("subset*/*.mhd")}
    uids = sorted(set(cands.seriesuid) & set(mhds))[: limit or None]
    for uid in uids:
        img, sp, org = read_mhd(mhds[uid])
        iso, iso_sp = resample_volume(img, sp, target_spacing)
        c = cands[cands.seriesuid == uid]
        pos, neg = c[c["class"] == 1], c[c["class"] == 0]
        neg = neg.iloc[rng.permutation(len(neg))[:neg_per_scan]]
        for i, r in pd.concat([pos, neg]).iterrows():
            vox = world_to_voxel((r.coordX, r.coordY, r.coordZ), org, sp) * np.asarray(sp) / np.asarray(iso_sp)
            cid = f"{uid}_{i}"
            np.savez_compressed(out_dir / "cand" / f"{cid}.npz",
                                hu=extract_patch(iso, vox, patch_vox).astype(np.float16))
            rows.append({"lesion_id": cid, "patient_id": pmap.get(uid, uid), "label": int(r["class"]),
                         "dataset": "LUNA16", "patch_path": f"cand/{cid}.npz"})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "candidates.csv", index=False)
    return df

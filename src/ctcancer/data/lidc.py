"""LIDC-IDRI -> lesion-level table + cached isotropic patches (uses `pylidc`).

Label definition (documented, configurable): each physical nodule is the cluster of reader
annotations (pylidc `cluster_annotations`). Its *radiologist-suspicion* label is derived from
the mean of the readers' 1–5 malignancy ratings:
    mean < 3  -> 0 ("likely benign" by radiologist consensus)
    mean > 3  -> 1 ("suspicious for malignancy")
    mean == 3 -> excluded (indeterminate)
Nodules annotated by fewer than `min_readers` readers are excluded.
This is a widely used *proxy* label. It is NOT biopsy-proven malignancy, and the app words
the output accordingly ("radiologist-suspicion-calibrated malignancy probability").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..preprocessing.ct import extract_patch, resample_mask, resample_volume

log = get_logger(__name__)


def _pylidc_compat() -> None:
    """pylidc 0.2.3 predates numpy 1.24 and Python 3.12. Restore what it still relies on:
    the numpy aliases np.int / np.float / np.bool (they always meant the builtins) and
    configparser.SafeConfigParser (removed in Python 3.12; it was an alias of ConfigParser)."""
    import configparser

    for name, typ in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, typ)
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser


def malignancy_label(ratings: list[int], rule: str = "mean_exclude_3") -> int | None:
    m = float(np.mean(ratings))
    if rule == "mean_exclude_3":
        if m < 3:
            return 0
        if m > 3:
            return 1
        return None
    if rule == "median_ge_3":  # looser variant (keeps more nodules, noisier)
        return int(np.median(ratings) >= 3)
    raise ValueError(rule)


def build_lidc_table(out_dir: str | Path, target_spacing=(1.0, 1.0, 1.0), patch_vox: int = 64,
                     min_readers: int = 2, rule: str = "mean_exclude_3", limit: int | None = None,
                     dicom_root: str | Path | None = None) -> pd.DataFrame:
    """Iterate all LIDC scans, cluster annotations, cache patches. Requires a configured ~/.pylidcrc."""
    _pylidc_compat()
    import pylidc as pl
    from pylidc.utils import consensus

    out_dir = Path(out_dir)
    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    rows = []
    all_scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id, pl.Scan.id).all()
    # Only scans whose patient folder was actually downloaded (pylidc's own order is not by patient).
    scans = [s for s in all_scans if dicom_root is None or (Path(dicom_root) / s.patient_id).is_dir()]
    log.info("LIDC: %d of %d annotated scans have a downloaded patient folder", len(scans), len(all_scans))
    if not scans:
        raise RuntimeError(f"No LIDC-IDRI-xxxx patient folders found in {dicom_root}. "
                           "Check the folder and the path in your pylidc settings file.")
    if limit:
        scans = scans[:limit]
    n_fail, first_err = 0, None
    for si, scan in enumerate(scans):
        try:
            vol_yxz = scan.to_volume(verbose=False).astype(np.float32)  # pylidc returns (rows, cols, slices) in HU
        except Exception as exc:  # missing files etc.
            n_fail += 1
            first_err = first_err or f"{type(exc).__name__}: {exc}"
            log.warning("skip scan %s (%s): %s: %s", scan.id, scan.patient_id, type(exc).__name__, exc)
            continue
        vol = np.transpose(vol_yxz, (2, 0, 1))  # -> (z, y, x), z ascending (inferior -> superior)
        spacing = (float(scan.slice_spacing), float(scan.pixel_spacing), float(scan.pixel_spacing))
        iso, iso_sp = resample_volume(vol, spacing, target_spacing, order=1)
        for ni, anns in enumerate(scan.cluster_annotations()):
            if len(anns) < min_readers:
                continue
            ratings = [a.malignancy for a in anns]
            label = malignancy_label(ratings, rule)
            cmask_yxz, cbbox, _ = consensus(anns, clevel=0.5, pad=[(0, 0), (0, 0), (0, 0)])
            full = np.zeros(vol_yxz.shape, bool)
            full[cbbox] = cmask_yxz
            mask = np.transpose(full, (2, 0, 1))
            mask_iso = resample_mask(mask, spacing, target_spacing)
            if not mask_iso.any():
                continue
            c = np.argwhere(mask_iso).mean(0)
            lesion_id = f"LIDC_{scan.patient_id}_{scan.id}_{ni}"
            np.savez_compressed(out_dir / "patches" / f"{lesion_id}.npz",
                                hu=extract_patch(iso, c, patch_vox).astype(np.float16),
                                mask=extract_patch(mask_iso.astype(np.uint8), c, patch_vox, fill=0).astype(bool),
                                spacing=np.array(iso_sp, np.float32))
            rows.append({
                "lesion_id": lesion_id, "patient_id": scan.patient_id, "dataset": "LIDC-IDRI",
                "n_readers": len(anns), "malignancy_mean": float(np.mean(ratings)),
                "malignancy_ratings": ";".join(map(str, ratings)),
                "label": label, "diameter_mm_pylidc": float(np.mean([a.diameter for a in anns])),
                "patch_path": f"patches/{lesion_id}.npz",
            })
        if si % 50 == 0:
            log.info("LIDC: %d/%d scans, %d nodules", si + 1, len(scans), len(rows))
    if not rows:
        raise RuntimeError(f"No LIDC nodules could be prepared ({n_fail} of {len(scans)} scans failed to load). "
                           f"First error: {first_err}")
    if n_fail:
        log.warning("LIDC: %d of %d scans could not be loaded and were skipped", n_fail, len(scans))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "lesions.csv", index=False)
    return df

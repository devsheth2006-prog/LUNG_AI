"""NSCLC-Radiomics / NSCLC-Radiogenomics: CT + tumour segmentation -> primary-tumour patches.

Tumour masks are read from DICOM-SEG objects (Lung1 ships GTV-1 as SEG; Radiogenomics ships
SEG for 144 subjects). Each SEG frame is matched to a CT slice by its ImagePositionPatient z.
Patients without a segmentation are skipped (and counted) — masks are never synthesised.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io.loaders import load_dicom_series
from ..logging_utils import get_logger
from ..preprocessing.ct import extract_patch, resample_mask, resample_volume

log = get_logger(__name__)


def _fg_item(fg, shared, name: str):
    """First item of a functional-group macro, from the per-frame group or else the shared group.
    (NSCLC-Radiomics SEGs store SegmentIdentification per frame; 3D-Slicer SEGs in
    NSCLC-Radiogenomics may store it once in the SharedFunctionalGroupsSequence.)"""
    for g in (fg, shared):
        if g is not None and name in g:
            return g[name].value[0]
    return None


def read_seg_mask(seg_path: Path, ct_positions_z: np.ndarray, ct_shape, segment_label: str = "GTV") -> np.ndarray | None:
    import pydicom

    ds = pydicom.dcmread(str(seg_path))
    if getattr(ds, "Modality", "") != "SEG":
        return None
    seg_numbers = {int(s.SegmentNumber): str(getattr(s, "SegmentLabel", "")) for s in ds.SegmentSequence}
    wanted = {n for n, lab in seg_numbers.items() if segment_label.lower() in lab.lower()} or set(seg_numbers)
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    if frames.shape[1:] != tuple(ct_shape[1:]):
        log.warning("SEG frame size %s differs from CT slice size %s: skipped", frames.shape[1:], tuple(ct_shape[1:]))
        return None
    shared = ds.SharedFunctionalGroupsSequence[0] if "SharedFunctionalGroupsSequence" in ds else None
    mask = np.zeros(ct_shape, bool)
    for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        sid = _fg_item(fg, shared, "SegmentIdentificationSequence")
        n = int(sid.ReferencedSegmentNumber) if sid is not None else min(seg_numbers)
        if n not in wanted:
            continue
        pos = _fg_item(fg, shared, "PlanePositionSequence")
        if pos is None:
            continue
        z = float(pos.ImagePositionPatient[2])
        k = int(np.argmin(np.abs(ct_positions_z - z)))
        if abs(ct_positions_z[k] - z) < 1.0:
            mask[k] |= frames[i].astype(bool)
    return mask if mask.any() else None


def _seg_referenced_series(seg_path: Path, files: list[Path]) -> str | None:
    """SeriesInstanceUID of the CT that a DICOM-SEG was drawn on.

    Some patients also have other CT series (e.g. the CT part of a PET/CT), so the largest series
    is not always the one the tumour mask belongs to."""
    import pydicom

    try:
        ds = pydicom.dcmread(str(seg_path), stop_before_pixels=True)
        if "ReferencedSeriesSequence" in ds:
            return str(ds.ReferencedSeriesSequence[0].SeriesInstanceUID)
        sop = str(ds.PerFrameFunctionalGroupsSequence[0].DerivationImageSequence[0]
                  .SourceImageSequence[0].ReferencedSOPInstanceUID)
    except Exception:
        return None
    for f in files:  # fall back: find the CT slice the first SEG frame was derived from
        try:
            d = pydicom.dcmread(str(f), stop_before_pixels=True)
            if str(getattr(d, "SOPInstanceUID", "")) == sop:
                return str(d.SeriesInstanceUID)
        except Exception:
            continue
    return None


def build_nsclc_table(root: str | Path, labels: pd.DataFrame, out_dir: str | Path, dataset: str,
                      target_spacing=(1.0, 1.0, 1.0), patch_vox: int = 96) -> pd.DataFrame:
    """For each patient folder: load CT series + SEG, cache a primary-tumour patch and the full mask stats."""
    import pydicom

    root, out_dir = Path(root), Path(out_dir)
    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for pid in labels.patient_id:
        pdir = root / pid
        if not pdir.exists():
            skipped += 1
            continue
        files = [p for p in pdir.rglob("*.dcm")]
        segs = [p for p in files if getattr(pydicom.dcmread(str(p), stop_before_pixels=True), "Modality", "") == "SEG"]
        if not segs:
            skipped += 1
            continue
        ref_uid = _seg_referenced_series(segs[0], files)
        try:
            vol = load_dicom_series(pdir, series_uid=ref_uid)
        except Exception as exc:
            log.warning("CT load failed (%s)", type(exc).__name__)
            skipped += 1
            continue
        nz = vol.shape[0]
        positions = vol.origin[0] + np.arange(nz) * vol.spacing[0]
        try:
            mask = read_seg_mask(segs[0], positions, vol.shape)
        except Exception as exc:  # unusual SEG encoding: skip this patient
            log.warning("%s: could not read SEG (%s: %s) - skipped", pid, type(exc).__name__, exc)
            mask = None
        if mask is None:
            skipped += 1
            continue
        iso, iso_sp = resample_volume(vol.array, vol.spacing, target_spacing)
        miso = resample_mask(mask, vol.spacing, target_spacing)
        c = np.argwhere(miso).mean(0)
        lid = f"{dataset}_{pid}"
        np.savez_compressed(out_dir / "patches" / f"{lid}.npz",
                            hu=extract_patch(iso, c, patch_vox).astype(np.float16),
                            mask=extract_patch(miso.astype(np.uint8), c, patch_vox, fill=0).astype(bool),
                            spacing=np.array(iso_sp, np.float32))
        rows.append({"lesion_id": lid, "patient_id": pid, "dataset": dataset,
                     "gtv_volume_cm3": float(miso.sum() * np.prod(iso_sp) / 1000),
                     "gtv_exceeds_patch": bool(np.ptp(np.argwhere(miso), 0).max() >= patch_vox),
                     "patch_path": f"patches/{lid}.npz"})
    log.info("%s: %d tumours cached, %d patients skipped (no CT/SEG)", dataset, len(rows), skipped)
    df = pd.DataFrame(rows).merge(labels, on="patient_id", how="left")
    df.to_csv(out_dir / "lesions.csv", index=False)
    return df

"""Loaders for DICOM series, NIfTI/MetaImage and PNG/JPEG (demonstration only).

DICOM handling is written out explicitly (rather than hidden in a library call) because
the conversion steps matter scientifically:

1. keep only CT image objects of one series (the one with the most slices);
2. order slices by position along the slice normal (ImagePositionPatient · normal),
   falling back to InstanceNumber — file names and acquisition order are unreliable;
3. detect missing slices from gaps in the slice positions;
4. convert stored values to Hounsfield Units: HU = stored * RescaleSlope + RescaleIntercept,
   after replacing padding values (outside the scan FOV) with air (-1024 HU).
"""
from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ..logging_utils import get_logger
from .validation import ScanValidationError, safe_extract_zip, sanity_check_volume, validate_upload
from .volume import SAFE_METADATA_KEYS, CTVolume, pseudonymise

log = get_logger(__name__)
CT_IMAGE_STORAGE = {"1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.2.1"}


# --------------------------------------------------------------------------- DICOM
def _iter_dicom_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from (q for q in p.rglob("*") if q.is_file())
        elif p.is_file():
            yield p


def load_dicom_series(paths: str | Path | Iterable[str | Path], series_uid: str | None = None) -> CTVolume:
    """Load a single CT series from a directory, a file, or a list of files.

    If `series_uid` is given and present, that series is used (e.g. the CT a SEG refers to);
    otherwise the series with the most slices is used."""
    import pydicom
    from pydicom.errors import InvalidDicomError

    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = list(_iter_dicom_files(Path(p) for p in paths))
    series: dict[str, list] = defaultdict(list)
    n_bad = 0
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if "PixelData" not in ds:
                continue
            sop = str(getattr(ds, "SOPClassUID", ""))
            modality = str(getattr(ds, "Modality", ""))
            if sop and sop not in CT_IMAGE_STORAGE and modality != "CT":
                continue
            series[str(getattr(ds, "SeriesInstanceUID", "unknown"))].append(ds)
        except (InvalidDicomError, AttributeError, OSError, ValueError):
            n_bad += 1
    if not series:
        raise ScanValidationError(
            "No readable CT image slices were found. The file(s) may be corrupted, may not be "
            "DICOM, or may contain a non-CT modality (e.g. RTSTRUCT, SEG, PET)."
        )
    if series_uid and series_uid in series:
        uid, slices = series_uid, series[series_uid]
    else:
        uid, slices = max(series.items(), key=lambda kv: len(kv[1]))
    warnings: list[str] = []
    if n_bad:
        warnings.append(f"{n_bad} unreadable/corrupted file(s) were skipped.")
    if len(series) > 1:
        how = "the requested one" if uid == series_uid else "the largest"
        warnings.append(f"{len(series)} series found; using {how} ({len(slices)} slices).")

    # ---- slice ordering along the slice normal
    try:
        iop = np.array(slices[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(iop[:3], iop[3:])
        if abs(normal[2]) < 0.8:
            warnings.append("Series is not axial; it is processed as stored (results may be unreliable).")
        positions = np.array([float(np.dot(normal, np.array(s.ImagePositionPatient, float))) for s in slices])
    except (AttributeError, TypeError, ValueError):
        warnings.append("ImagePositionPatient missing; slices ordered by InstanceNumber.")
        positions = np.array([float(getattr(s, "InstanceNumber", i)) for i, s in enumerate(slices)])
    order = np.argsort(positions)
    slices = [slices[i] for i in order]
    positions = positions[order]

    # remove exact duplicate positions (repeated acquisitions)
    keep = np.concatenate([[True], np.diff(positions) > 1e-3])
    if (~keep).any():
        warnings.append(f"{int((~keep).sum())} duplicate slice position(s) removed.")
    slices = [s for s, k in zip(slices, keep) if k]
    positions = positions[keep]

    # ---- spacing + missing-slice detection
    if len(positions) > 1:
        gaps = np.diff(positions)
        dz = float(np.median(gaps))
        n_missing = int(np.sum(np.round(gaps / dz) - 1))
        if n_missing > 0:
            warnings.append(f"Approximately {n_missing} missing slice(s) detected (gaps in slice positions); "
                            "gaps are filled by interpolation during resampling.")
    else:
        dz = float(getattr(slices[0], "SliceThickness", 1.0) or 1.0)
    try:
        dy, dx = (float(v) for v in slices[0].PixelSpacing)
    except (AttributeError, TypeError, ValueError):
        dy = dx = 1.0
        warnings.append("PixelSpacing missing; assuming 1 mm (measurements unreliable).")

    # ---- pixel data -> HU
    arrays = []
    for s in slices:
        try:
            arr = s.pixel_array.astype(np.float32)
        except Exception as exc:  # pixel decoding errors are library-specific
            raise ScanValidationError(f"Could not decode pixel data ({type(exc).__name__}). "
                                      "The series may be compressed with an unsupported codec.") from exc
        pad = getattr(s, "PixelPaddingValue", None)
        slope = float(getattr(s, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(s, "RescaleIntercept", 0.0) or 0.0)
        hu = arr * slope + intercept
        if pad is not None:
            hu[arr == float(pad)] = -1024.0
        hu[hu < -1024] = -1024.0  # out-of-FOV values (often -2000/-3024)
        arrays.append(hu)
    shapes = {a.shape for a in arrays}
    if len(shapes) > 1:
        raise ScanValidationError(f"Slices have inconsistent sizes {shapes}; the series is corrupted.")
    volume = np.stack(arrays).astype(np.float32)

    first = slices[0]
    meta = {k: _to_builtin(getattr(first, k)) for k in SAFE_METADATA_KEYS if hasattr(first, k)}
    origin = tuple(float(v) for v in getattr(first, "ImagePositionPatient", (0, 0, 0)))[::-1]
    warnings += sanity_check_volume(volume, is_hu=True)
    log.info("Loaded DICOM series: %d slices, spacing=(%.2f, %.2f, %.2f)", volume.shape[0], dz, dy, dx)
    return CTVolume(volume, (dz, dy, dx), origin, True, "dicom", pseudonymise(uid), meta, warnings)


def _to_builtin(v):
    try:
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            return [float(x) for x in v]
        return float(v) if isinstance(v, (int, float)) else str(v)
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------- NIfTI / MHD
def load_sitk_image(path: str | Path, assume_hu: bool = True) -> CTVolume:
    """Load NIfTI / MetaImage / NRRD with SimpleITK and reorient to LPS."""
    import SimpleITK as sitk

    try:
        img = sitk.ReadImage(str(path))
    except RuntimeError as exc:
        raise ScanValidationError("The volume file could not be read (corrupted or unsupported).") from exc
    if img.GetDimension() == 4:
        raise ScanValidationError("4-D images (time series) are not supported.")
    if img.GetDimension() != 3:
        raise ScanValidationError(f"Expected a 3-D image, got {img.GetDimension()}-D.")
    img = sitk.DICOMOrient(img, "LPS")
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (z, y, x)
    sx, sy, sz = img.GetSpacing()
    ox, oy, oz = img.GetOrigin()
    warnings = sanity_check_volume(arr, is_hu=assume_hu)
    fmt = "nifti" if ".nii" in str(path).lower() else "mhd"
    return CTVolume(arr, (sz, sy, sx), (oz, oy, ox), assume_hu, fmt, pseudonymise(Path(path).name), {}, warnings)


# --------------------------------------------------------------------------- PNG/JPEG
def load_raster_image(path: str | Path) -> CTVolume:
    """Load a PNG/JPEG *for demonstration only*.

    A screenshot/export of a CT slice has already been windowed and quantised to 8 bits,
    so Hounsfield Units are lost and pixel spacing is unknown. We therefore mark the volume
    as non-quantitative: HU-based steps and mm measurements are disabled downstream.
    Colour images are converted to grayscale — replicating gray into RGB adds no information.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ScanValidationError("The image file could not be decoded.")
    arr = img.astype(np.float32)[None]
    if min(arr.shape[1:]) < 64:
        raise ScanValidationError("Image is too small.")
    warnings = ["PNG/JPEG input: Hounsfield Units and pixel spacing are unavailable; "
                "quantitative measurements and HU-dependent models are disabled (demonstration mode)."]
    return CTVolume(arr, (1.0, 1.0, 1.0), (0, 0, 0), False, "raster", pseudonymise(Path(path).name),
                    {"spacing_known": False}, warnings)


# --------------------------------------------------------------------------- dispatcher
def load_any(path: str | Path, max_mb: float = 500) -> CTVolume:
    """Validate and load any supported upload (file, zip of DICOM, or directory)."""
    path = Path(path)
    if path.is_dir():
        return load_dicom_series(path)
    fmt = validate_upload(path, max_mb=max_mb)
    if fmt == "zip":
        tmp = Path(tempfile.mkdtemp(prefix="ctcancer_"))
        safe_extract_zip(path, tmp)
        nii = [p for p in tmp.rglob("*") if p.name.lower().endswith((".nii", ".nii.gz", ".mhd", ".mha"))]
        return load_sitk_image(nii[0]) if nii else load_dicom_series(tmp)
    if fmt in {"dicom", "dicom-maybe"}:
        vol = load_dicom_series(path)
        return vol
    if fmt in {"nifti", "mhd", "nrrd"}:
        return load_sitk_image(path)
    if fmt in {"png", "jpeg"}:
        return load_raster_image(path)
    raise ScanValidationError(f"Unsupported format: {fmt}")

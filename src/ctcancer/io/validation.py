"""Upload validation: size limits, format sniffing, safe archive extraction, sanity checks."""
from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import numpy as np

ALLOWED_SUFFIXES = {".dcm", ".dicom", ".nii", ".nii.gz", ".mhd", ".mha", ".nrrd",
                    ".png", ".jpg", ".jpeg", ".zip", ""}


class ScanValidationError(ValueError):
    """Raised with a user-facing message when an uploaded scan is unusable."""


def full_suffix(name: str) -> str:
    n = name.lower()
    if n.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(n).suffix


def sniff_format(path: Path) -> str:
    """Identify a file by content, not by extension."""
    with open(path, "rb") as f:
        head = f.read(352)
    if len(head) >= 132 and head[128:132] == b"DICM":
        return "dicom"
    if head[:4] == b"PK\x03\x04":
        return "zip"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:2] == b"\x1f\x8b":
        try:
            with gzip.open(path, "rb") as g:
                inner = g.read(348)
            if len(inner) >= 4 and int.from_bytes(inner[:4], "little") == 348:
                return "nifti"
        except OSError:
            pass
        return "gzip-unknown"
    if len(head) >= 4 and int.from_bytes(head[:4], "little") == 348:
        return "nifti"
    if head.startswith(b"ObjectType") or b"ElementDataFile" in head or head.startswith(b"NDims"):
        return "mhd"
    if head.startswith(b"NRRD"):
        return "nrrd"
    # DICOM files without the 128-byte preamble (old ACR-NEMA style)
    if path.suffix.lower() in {".dcm", ".dicom", ""}:
        return "dicom-maybe"
    return "unknown"


def validate_upload(path: str | Path, max_mb: float = 500) -> str:
    """Check size, extension and content; return the sniffed format."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise ScanValidationError("The uploaded file is empty or missing.")
    size_mb = path.stat().st_size / 1e6
    if size_mb > max_mb:
        raise ScanValidationError(f"File is {size_mb:.0f} MB; the limit is {max_mb:.0f} MB.")
    if full_suffix(path.name) not in ALLOWED_SUFFIXES:
        raise ScanValidationError(
            f"Unsupported file type '{full_suffix(path.name)}'. Upload DICOM (.dcm or a .zip of a "
            "series), NIfTI (.nii/.nii.gz), MetaImage (.mhd/.mha) or PNG/JPEG for demonstration."
        )
    fmt = sniff_format(path)
    if fmt in {"unknown", "gzip-unknown"}:
        raise ScanValidationError("The file content does not match any supported medical image format.")
    return fmt


def safe_extract_zip(zip_path: str | Path, dest: str | Path, max_files: int = 5000,
                     max_total_mb: float = 2000) -> Path:
    """Extract a zip while blocking path traversal ('zip slip') and decompression bombs."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ScanValidationError("The zip archive is corrupted.") from exc
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > max_files:
            raise ScanValidationError(f"Archive contains {len(infos)} files (limit {max_files}).")
        if sum(i.file_size for i in infos) / 1e6 > max_total_mb:
            raise ScanValidationError("Archive expands beyond the allowed size.")
        for info in infos:
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest)):
                raise ScanValidationError("Archive contains unsafe paths.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
    return dest


def sanity_check_volume(array: np.ndarray, is_hu: bool) -> list[str]:
    """Return warnings (not errors) about suspicious volumes."""
    warnings: list[str] = []
    if array.ndim != 3:
        raise ScanValidationError(f"Expected a 3-D volume, got {array.ndim} dimensions.")
    if min(array.shape[1:]) < 64:
        raise ScanValidationError(f"In-plane size {array.shape[1:]} is too small for CT analysis.")
    if not np.isfinite(array).all():
        raise ScanValidationError("The image contains NaN/Inf values (corrupted data).")
    if is_hu:
        lo, hi = np.percentile(array, [1, 99])
        if lo > -500 or hi < 0:
            warnings.append(
                f"Intensity range ({lo:.0f}..{hi:.0f}) is atypical for chest CT in HU; "
                "the scan may not be a chest CT or rescale tags may be wrong."
            )
    if array.shape[0] == 1:
        warnings.append("Single slice: volumetric measurements and 3-D analysis are unavailable.")
    elif array.shape[0] < 30:
        warnings.append(f"Only {array.shape[0]} slices: coverage may be incomplete.")
    return warnings

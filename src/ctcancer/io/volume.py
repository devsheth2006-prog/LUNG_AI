"""The CTVolume container used by every downstream stage."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# DICOM attributes that are safe to keep (technical, non-identifying).
SAFE_METADATA_KEYS = (
    "Modality", "Manufacturer", "ManufacturerModelName", "SliceThickness", "KVP",
    "ConvolutionKernel", "PixelSpacing", "ContrastBolusAgent", "BodyPartExamined",
    "PatientSex", "PatientAge",  # coarse demographics, used only if the user opts in
)


@dataclass
class CTVolume:
    """A CT volume in Hounsfield Units, array order (z, y, x).

    Orientation convention: axial slices; column index increases towards the
    patient's LEFT and row index towards POSTERIOR (DICOM LPS). Displayed without
    flipping, this gives the standard radiological view (patient right on screen left).
    """

    array: np.ndarray                      # float32, HU if is_hu else arbitrary intensities
    spacing: tuple[float, float, float]    # (z, y, x) in mm
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_hu: bool = True
    source_format: str = "unknown"         # dicom | nifti | mhd | raster | array
    scan_id: str = "anonymous"
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.array.shape)  # type: ignore[return-value]

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def is_volumetric(self) -> bool:
        return self.array.shape[0] > 1

    @property
    def quantitative(self) -> bool:
        """True when physical measurements (mm, HU) are meaningful."""
        return self.is_hu and self.metadata.get("spacing_known", True)

    def summary(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "format": self.source_format,
            "shape_zyx": list(self.shape),
            "spacing_mm_zyx": [round(s, 3) for s in self.spacing],
            "extent_mm_zyx": [round(n * s, 1) for n, s in zip(self.shape, self.spacing)],
            "hounsfield_units": self.is_hu,
            "hu_range": [float(np.percentile(self.array, 0.5)), float(np.percentile(self.array, 99.5))],
            "warnings": list(self.warnings),
            "metadata": {k: v for k, v in self.metadata.items() if k in SAFE_METADATA_KEYS},
        }


def pseudonymise(identifier: str, salt: str = "ctcancer") -> str:
    """One-way pseudonymous scan identifier (never store the raw UID/patient ID)."""
    return "SCAN-" + hashlib.sha256((salt + identifier).encode()).hexdigest()[:10].upper()

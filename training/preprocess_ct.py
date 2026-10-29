"""Preprocess a single CT (DICOM folder / zip / NIfTI) and save the result for inspection.

python training/preprocess_ct.py --input path/to/series_dir --out outputs/preprocessed
Writes <scan_id>.npz (HU volume, lung-window volume, spacing) and a NIfTI copy you can open in
3D Slicer / ITK-SNAP to verify orientation and resampling.
"""
import _bootstrap  # noqa: F401

from pathlib import Path

import numpy as np

from ctcancer.config import cli_parser, load_config
from ctcancer.io import load_any
from ctcancer.logging_utils import get_logger
from ctcancer.preprocessing import preprocess_volume
from ctcancer.segmentation import classical_lung_mask

log = get_logger("preprocess_ct")


def main():
    p = cli_parser(__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="outputs/preprocessed")
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    vol = load_any(a.input)
    for w in vol.warnings:
        log.warning(w)
    pre = preprocess_volume(vol, cfg["preprocessing"])
    lung = classical_lung_mask(pre.hu, pre.spacing, pre.is_hu)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{vol.scan_id}.npz", hu=pre.hu.astype(np.int16), norm=pre.norm.astype(np.float16),
                        lung=lung, spacing=np.array(pre.spacing))
    try:
        import SimpleITK as sitk

        img = sitk.GetImageFromArray(pre.hu.astype(np.int16))
        img.SetSpacing(tuple(float(s) for s in pre.spacing[::-1]))
        sitk.WriteImage(img, str(out / f"{vol.scan_id}.nii.gz"))
        m = sitk.GetImageFromArray(lung.astype(np.uint8))
        m.CopyInformation(img)
        sitk.WriteImage(m, str(out / f"{vol.scan_id}_lung.nii.gz"))
    except ImportError:
        pass
    log.info("Steps: %s", " | ".join(pre.steps))
    log.info("Saved to %s (%s)", out, vol.scan_id)


if __name__ == "__main__":
    main()

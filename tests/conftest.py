import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ctcancer.phantom import make_phantom  # noqa: E402


@pytest.fixture(scope="session")
def phantom():
    return make_phantom()


@pytest.fixture(scope="session")
def phantom_nifti(tmp_path_factory, phantom):
    import SimpleITK as sitk

    vol, _ = phantom
    img = sitk.GetImageFromArray(vol.array.astype(np.int16))
    img.SetSpacing(tuple(float(s) for s in vol.spacing[::-1]))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))  # LPS identity
    p = tmp_path_factory.mktemp("nii") / "phantom.nii.gz"
    sitk.WriteImage(img, str(p))
    return p


@pytest.fixture()
def empty_models(tmp_path):
    """A models dir with nothing trained — guarantees tests never see real/smoke models."""
    d = tmp_path / "models"
    d.mkdir()
    return d


def write_dicom_series(folder: Path, vol_hu: np.ndarray, spacing=(2.5, 0.8, 0.8), slope=1.0, intercept=-1024.0,
                       drop: tuple[int, ...] = (), shuffle: bool = True, pad_value: int | None = None):
    """Write a CT series with real DICOM encoding (stored = (HU - intercept)/slope)."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    folder.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    order = np.random.default_rng(0).permutation(vol_hu.shape[0]) if shuffle else range(vol_hu.shape[0])
    for i in order:
        if i in drop:
            continue
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"f{i:04d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.Modality, ds.SeriesInstanceUID, ds.StudyInstanceUID = "CT", series_uid, generate_uid()
        ds.PatientName, ds.PatientID = "SECRET^NAME", "SECRET-ID-123"
        ds.Rows, ds.Columns = vol_hu.shape[1:]
        ds.PixelSpacing = [spacing[1], spacing[2]]
        ds.SliceThickness = spacing[0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i * spacing[0])]
        ds.InstanceNumber = int(vol_hu.shape[0] - i)  # deliberately misleading order
        ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        stored = np.round((vol_hu[i] - intercept) / slope).astype(np.int16)
        if pad_value is not None:
            stored[:2, :2] = pad_value
            ds.PixelPaddingValue = pad_value
        ds.PixelData = stored.tobytes()
        ds.save_as(str(folder / f"f{i:04d}.dcm"), enforce_file_format=True)
    return folder

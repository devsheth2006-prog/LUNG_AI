import zipfile

import numpy as np
import pytest

from conftest import write_dicom_series

from ctcancer.io import ScanValidationError, load_any, load_dicom_series, safe_extract_zip, validate_upload


def _vol():
    rng = np.random.default_rng(1)
    v = np.full((12, 96, 96), -1000.0)
    v[:, 20:76, 20:76] = 40
    v[:, 30:66, 25:45] = -850
    v[:, 30:66, 51:71] = -850
    return v + rng.normal(0, 1, v.shape)


def test_dicom_hu_conversion_and_ordering(tmp_path):
    v = _vol()
    v[3] += 100  # marker slice to verify ordering
    write_dicom_series(tmp_path / "s", v, slope=2.0, intercept=-1024)
    vol = load_dicom_series(tmp_path / "s")
    assert vol.shape == v.shape
    assert vol.is_hu
    # slice order follows ImagePositionPatient, not file order / InstanceNumber
    assert np.argmax(vol.array.mean(axis=(1, 2))) == 3
    assert np.abs(vol.array[5, 50, 50] - v[5, 50, 50]) <= 2.0  # slope 2 quantisation
    assert vol.spacing == pytest.approx((2.5, 0.8, 0.8))


def test_missing_slice_and_padding(tmp_path):
    write_dicom_series(tmp_path / "s", _vol(), drop=(6,), pad_value=-2000)
    vol = load_dicom_series(tmp_path / "s")
    assert any("missing slice" in w for w in vol.warnings)
    assert vol.array[0, 0, 0] == -1024.0  # padding mapped to air


def test_patient_identifiers_not_retained(tmp_path):
    write_dicom_series(tmp_path / "s", _vol())
    vol = load_dicom_series(tmp_path / "s")
    blob = repr(vol.summary()) + repr(vol.metadata)
    assert "SECRET" not in blob
    assert vol.scan_id.startswith("SCAN-")


def test_corrupted_files_skipped_and_all_corrupted_rejected(tmp_path):
    write_dicom_series(tmp_path / "s", _vol())
    (tmp_path / "s" / "broken.dcm").write_bytes(b"\0" * 128 + b"DICMgarbage")
    vol = load_dicom_series(tmp_path / "s")
    assert vol.shape[0] == 12
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "x.dcm").write_bytes(b"not a dicom at all")
    with pytest.raises(ScanValidationError):
        load_dicom_series(bad)


def test_unsupported_and_empty_files(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(ScanValidationError, match="Unsupported"):
        validate_upload(p)
    e = tmp_path / "empty.nii"
    e.write_bytes(b"")
    with pytest.raises(ScanValidationError, match="empty"):
        validate_upload(e)
    big = tmp_path / "big.nii"
    big.write_bytes(b"\0" * 2_000_000)
    with pytest.raises(ScanValidationError, match="limit"):
        validate_upload(big, max_mb=1)


def test_nifti_roundtrip(phantom_nifti, phantom):
    vol = load_any(phantom_nifti)
    assert vol.shape == phantom[0].shape
    assert vol.spacing == pytest.approx(phantom[0].spacing)
    assert vol.is_hu


def test_png_is_marked_non_quantitative(tmp_path):
    import cv2

    img = (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8)
    p = tmp_path / "slice.png"
    cv2.imwrite(str(p), img)
    vol = load_any(p)
    assert not vol.is_hu and not vol.quantitative
    assert any("Hounsfield" in w for w in vol.warnings)


def test_zip_slip_blocked(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../escape.txt", "x")
    with pytest.raises(ScanValidationError, match="unsafe"):
        safe_extract_zip(z, tmp_path / "out")


def test_zip_of_dicom_series(tmp_path):
    write_dicom_series(tmp_path / "s", _vol())
    z = tmp_path / "series.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in (tmp_path / "s").iterdir():
            zf.write(f, f"study/{f.name}")
    assert load_any(z).shape == (12, 96, 96)

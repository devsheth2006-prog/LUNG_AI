"""Integration: inference pipeline, registry honesty, report generation and API endpoints."""
import json

import numpy as np
import pytest

from ctcancer.config import load_config
from ctcancer.inference.pipeline import AnalysisPipeline
from ctcancer.inference.registry import task_status, write_card
from ctcancer.reporting.report import REPORT_DISCLAIMER, build_report


def _pipe(models_dir):
    return AnalysisPipeline(load_config(overrides=[f"paths.models_dir={models_dir}"]))


def test_pipeline_untrained_gives_no_probabilities(phantom, empty_models):
    vol, nod = phantom
    res, arr = _pipe(empty_models).run(vol)
    assert res.ok, res.error
    assert "classical" in res.lung["method"]
    assert len(res.lesions) >= 1
    idx = next(l for l in res.lesions if l["id"] == res.index_lesion)
    assert idx["malignancy"]["probability"] is None
    assert idx["detection"]["probability"] is None and "not a probability" in idx["detection"]["note"]
    assert idx["location"]["side"] == "right lung"
    assert idx["measurements"]["max_axial_diameter_mm"] == pytest.approx(14, abs=2.5)
    for k in ("subtype", "t_category", "n_involvement", "m_distant"):
        assert res.patient_predictions[k]["probabilities"] is None
        assert res.patient_predictions[k]["badge"] == "Model not trained"
    assert res.patient_predictions["organ_specific_metastasis"]["state"] == "unsupported"
    json.dumps(res.to_dict())  # serialisable


def test_pipeline_graceful_errors(tmp_path, empty_models):
    bad = tmp_path / "scan.nii.gz"
    bad.write_bytes(b"\x1f\x8b\x08\x00garbage")
    res, arr = _pipe(empty_models).run(bad)
    assert not res.ok and arr is None
    assert "Traceback" not in (res.error or "")
    missing = tmp_path / "nope.dcm"
    res, _ = _pipe(empty_models).run(missing)
    assert not res.ok


def test_pipeline_no_lung_found(empty_models):
    from ctcancer.io.volume import CTVolume

    v = CTVolume(np.full((20, 96, 96), 40.0, np.float32), (2.0, 1.0, 1.0))
    res, _ = _pipe(empty_models).run(v)
    assert res.ok and not res.lesions
    assert any("No lung region" in w for w in res.warnings)


def test_registry_states(tmp_path):
    assert task_status("malignancy", tmp_path).state == "not_trained"
    write_card("malignancy", {"architecture": "resnet18"}, tmp_path)
    assert task_status("malignancy", tmp_path).message.startswith("Model card found but weights")
    write_card("m_distant", {"refused": True, "reason": "too few"}, tmp_path)
    assert task_status("m_distant", tmp_path).state == "refused"
    assert task_status("organ_specific_metastasis", tmp_path).state == "unsupported"


def test_report_contains_disclaimer_and_no_fake_numbers(phantom, empty_models, tmp_path):
    res, arr = _pipe(empty_models).run(phantom[0])
    paths = build_report(res, arr, tmp_path)
    html = paths["html"].read_text()
    assert REPORT_DISCLAIMER in html
    assert "Model not trained" in html
    assert "data:image/png;base64" in html
    assert paths["pdf"].exists() and paths["pdf"].stat().st_size > 2000


def test_api_endpoints(phantom_nifti, tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import api.main as api

    api._pipeline = _pipe(tmp_path)
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/models").json()["malignancy"]["state"] == "not_trained"
    assert len(c.get("/datasets").json()["datasets"]) >= 4
    with open(phantom_nifti, "rb") as f:
        r = c.post("/analyze", files={"file": ("phantom.nii.gz", f, "application/gzip")},
                   data={"age": "60", "include_report": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and "report_html" in body and body["lesions"]
    assert "deep_embedding" not in body["lesions"][0]
    r = c.post("/analyze", files={"file": ("x.exe", b"MZ....", "application/octet-stream")})
    assert r.status_code == 415
    r = c.post("/analyze", files={"file": ("x.dcm", b"junk" * 50, "application/dicom")})
    assert r.status_code == 422

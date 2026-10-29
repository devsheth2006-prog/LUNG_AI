"""FastAPI inference service — lets the model backend run separately from the UI
(e.g. UI on Streamlit Cloud, inference on a GPU box / Render / Docker).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Uploads are written to a private temp dir, size-limited while streaming, processed locally and
deleted immediately. No patient identifiers are logged or returned.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from ctcancer import DISCLAIMER, __version__  # noqa: E402
from ctcancer.config import load_config  # noqa: E402
from ctcancer.data.catalog import UNSUPPORTED, dataset_table  # noqa: E402
from ctcancer.inference.pipeline import AnalysisPipeline  # noqa: E402
from ctcancer.inference.registry import all_statuses  # noqa: E402
from ctcancer.io.validation import ALLOWED_SUFFIXES, full_suffix  # noqa: E402
from ctcancer.reporting.report import build_html, render_figures  # noqa: E402

CFG = load_config(profile=os.getenv("CTC_PROFILE", "lite"))
MAX_MB = float(os.getenv("CTC_MAX_UPLOAD_MB", CFG.get_path("app.max_upload_mb", 500)))
app = FastAPI(title="CT Lung Cancer Research API", version=__version__,
              description="Research prototype — outputs are model-generated estimates, not diagnoses.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CTC_CORS_ORIGINS", "http://localhost:8501").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])
_pipeline: AnalysisPipeline | None = None


def pipeline() -> AnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AnalysisPipeline(CFG)
    return _pipeline


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "disclaimer": DISCLAIMER}


@app.get("/models")
def models():
    out = {}
    for k, s in all_statuses().items():
        card = {kk: vv for kk, vv in (s.card or {}).items() if kk not in ("history", "comparison")}
        out[k] = {"state": s.state, "badge": s.badge, "message": s.message, "card": card or None}
    return out


@app.get("/datasets")
def datasets():
    return {"datasets": dataset_table(), "unsupported": UNSUPPORTED}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), age: float | None = Form(None), sex: str | None = Form(None),
                  include_report: bool = Form(False)):
    name = Path(file.filename or "upload").name
    if full_suffix(name) not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Unsupported file type '{full_suffix(name)}'.")
    tmp = Path(tempfile.mkdtemp(prefix="ctc_api_"))
    try:
        dst = tmp / name
        size = 0
        with open(dst, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_MB * 1e6:
                    raise HTTPException(413, f"Upload exceeds {MAX_MB:.0f} MB.")
                f.write(chunk)
        clinical = {}
        if age is not None:
            if not 0 < age < 120:
                raise HTTPException(422, "age must be between 0 and 120")
            clinical["age"] = age
        if sex in ("male", "female"):
            clinical["sex_male"] = 1.0 if sex == "male" else 0.0
        res, arrays = pipeline().run(dst, clinical, keep_arrays=include_report)
        if not res.ok:
            raise HTTPException(422, res.error or "Analysis failed.")
        body = res.to_dict()
        for les in body["lesions"]:
            les.pop("deep_embedding", None)
        if include_report:
            body["report_html"] = build_html(res, render_figures(res, arrays))
        return body
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # never keep uploaded scans

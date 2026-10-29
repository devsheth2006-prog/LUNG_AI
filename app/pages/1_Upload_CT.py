import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from ui import nav_link, disclaimer, page

from ctcancer.config import load_config
from ctcancer.inference.pipeline import AnalysisPipeline
from ctcancer.phantom import make_phantom

page("Upload CT")


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return AnalysisPipeline(load_config())


def run_remote(api_url: str, path: Path, clinical: dict):
    """Split deployment: send the scan to the FastAPI inference service (no image arrays returned)."""
    import httpx

    from ctcancer.inference.pipeline import AnalysisResult

    data = {k: str(v) for k, v in {"age": clinical.get("age"),
                                   "sex": {1.0: "male", 0.0: "female"}.get(clinical.get("sex_male"))}.items() if v}
    try:
        with open(path, "rb") as f:
            r = httpx.post(f"{api_url.rstrip('/')}/analyze", files={"file": (path.name, f)}, data=data, timeout=600)
    except httpx.HTTPError as exc:
        return AnalysisResult(ok=False, error=f"Inference service unreachable ({type(exc).__name__}).")
    if r.status_code != 200:
        return AnalysisResult(ok=False, error=r.json().get("detail", "Analysis failed."))
    body = r.json()
    return AnalysisResult(**{k: v for k, v in body.items() if k in AnalysisResult.__dataclass_fields__})


cfg = load_config()
max_mb = cfg.get_path("app.max_upload_mb", 500)
st.markdown("Supported: **DICOM** (a `.zip` of one series, or a single `.dcm` slice), **NIfTI** (`.nii`, `.nii.gz`), "
            "**MetaImage** (`.mhd`+`.raw` zipped, `.mha`). PNG/JPEG are accepted for demonstration only — they have no "
            "Hounsfield Units or pixel spacing, so quantitative outputs and HU-based models are disabled for them.")

src = st.radio("Input", ["Upload a scan", "Synthetic software-test phantom"], horizontal=True)
up = None
if src == "Upload a scan":
    up = st.file_uploader("CT scan", type=["zip", "dcm", "nii", "gz", "mha", "png", "jpg", "jpeg"],
                          help=f"Maximum {max_mb} MB. Files are deleted after processing.")
else:
    st.warning("The phantom is a geometric object made of ellipsoids with textbook HU values. It is **not patient "
               "data** and exists only to demonstrate/test the software path. Any model output on it is meaningless.")

with st.expander("Optional clinical variables (used only by models trained with them)"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", 0, 110, value=None, placeholder="not provided")
    sex = c2.selectbox("Sex", ["not provided", "female", "male"])
    st.caption("Only variables that exist in the training cohorts are offered. Leave blank if unknown — missing values "
               "are handled explicitly, never guessed.")

run = st.button("Run analysis", type="primary", disabled=(src == "Upload a scan" and up is None))
if run:
    clinical = {}
    if age:
        clinical["age"] = float(age)
    if sex != "not provided":
        clinical["sex_male"] = 1.0 if sex == "male" else 0.0
    tmp = Path(tempfile.mkdtemp(prefix="ctc_ui_"))
    try:
        with st.status("Analysing…", expanded=True) as status:
            if src == "Upload a scan":
                if up.size > max_mb * 1e6:
                    st.error(f"File exceeds {max_mb} MB.")
                    st.stop()
                path = tmp / Path(up.name).name
                path.write_bytes(up.getbuffer())
                target = path
            else:
                target, _ = make_phantom()
            st.write("Validating and loading scan…")
            api_url = os.getenv("CTC_API_URL") or cfg.get_path("app.api_url")
            if api_url and src == "Upload a scan":
                res, arrays = run_remote(api_url, target, clinical), None
            else:
                res, arrays = get_pipeline().run(target, clinical)
            if res.ok:
                for s in res.preprocessing:
                    st.write("✓", s)
                st.write("✓ Lung segmentation —", res.lung.get("method"))
                st.write(f"✓ {len(res.lesions)} lesion candidate(s)")
                status.update(label=f"Done in {res.runtime_s:.1f} s", state="complete")
            else:
                status.update(label="Could not process this scan", state="error")
        st.session_state["result"], st.session_state["arrays"] = res, arrays
    finally:
        if not cfg.get_path("app.keep_uploads", False):
            shutil.rmtree(tmp, ignore_errors=True)

res = st.session_state.get("result")
if res is not None:
    if not res.ok:
        st.error(res.error)
    else:
        for w in res.warnings:
            st.warning(w)
        st.success("Scan processed. Continue with the pages in the sidebar.")
        c = st.columns(4)
        nav_link("pages/2_CT_Viewer.py", "CT viewer", "🩻", c[0])
        nav_link("pages/4_Cancer_Analysis.py", "Analysis dashboard", "📊", c[1])
        nav_link("pages/7_Explainability.py", "Explainability", "🔍", c[2])
        nav_link("pages/9_Report.py", "Report", "📄", c[3])
disclaimer()

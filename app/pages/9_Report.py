import tempfile
from pathlib import Path

import streamlit as st

from ui import disclaimer, page, require_scan

from ctcancer.reporting.report import REPORT_DISCLAIMER, build_report

page("Research Report")
res, A = require_scan()
st.markdown(f"**{REPORT_DISCLAIMER}**")
if st.button("Generate report", type="primary") or "report_paths" in st.session_state:
    if "report_paths" not in st.session_state or st.session_state.get("report_for") != res.scan["scan_id"]:
        out = Path(tempfile.mkdtemp(prefix="ctc_report_"))
        with st.spinner("Rendering report…"):
            st.session_state["report_paths"] = build_report(res, A, out)
            st.session_state["report_for"] = res.scan["scan_id"]
    paths = st.session_state["report_paths"]
    c = st.columns(3)
    c[0].download_button("Download HTML", paths["html"].read_bytes(), paths["html"].name, "text/html")
    if "pdf" in paths:
        c[1].download_button("Download PDF", paths["pdf"].read_bytes(), paths["pdf"].name, "application/pdf")
    import json

    c[2].download_button("Download JSON", json.dumps(res.to_dict(), indent=2), f"{res.scan['scan_id']}.json",
                         "application/json")
    st.iframe(paths["html"].read_text(encoding="utf-8"), height=1400)  # our own escaped HTML
disclaimer()

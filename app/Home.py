"""Streamlit entry point:  streamlit run app/Home.py"""
import pandas as pd
import streamlit as st

from ui import nav_link, chip, disclaimer, page

from ctcancer.data.catalog import UNSUPPORTED, dataset_table
from ctcancer.inference.registry import all_statuses
from ctcancer.risk.tasks import TASKS

page("AI-Assisted CT Lung Cancer Analysis", "Academic research prototype")
disclaimer()

st.markdown("""
#### What this platform does
Upload a chest CT and the pipeline runs **preprocessing → lung segmentation → nodule candidate detection →
lesion segmentation → radiomic & deep feature extraction → malignancy estimation → subtype / TNM / spread
risk estimation → Grad-CAM explainability → research report**. Each stage reports *how* it produced its
output — a trained and evaluated model, an untrained classical baseline, or "not available" — so nothing
is presented with more confidence than the evidence supports.
""")

c1, c2 = st.columns([1.05, 1])
with c1:
    st.markdown("##### Pipeline")
    steps = ["Validate upload (format, size, integrity)", "DICOM → Hounsfield Units, slice ordering, resampling",
             "Lung segmentation (U-Net or classical baseline)", "Nodule candidates + CNN false-positive reduction",
             "Lesion segmentation (U-Net)", "Radiomics (shape · first-order · GLCM) + CNN embedding",
             "Malignancy-suspicion probability (calibrated)", "Subtype · T · N · M estimates (where labels exist)",
             "Grad-CAM / Grad-CAM++ attention", "Structured HTML/PDF research report"]
    st.markdown("".join(f'<div class="step"><span class="i">{i:02d}</span>{s}</div>' for i, s in enumerate(steps, 1)),
                unsafe_allow_html=True)
with c2:
    st.markdown("##### Module status")
    statuses = all_statuses()
    rows = "".join(f'<div class="step"><span style="flex:1">{TASKS[k].title}</span>{chip(s.state)}</div>'
                   for k, s in statuses.items())
    st.markdown(rows, unsafe_allow_html=True)
    st.caption("Status is read from `models/<task>/model_card.json`, written only by the training scripts.")

st.markdown("##### Datasets and what they can (and cannot) supervise")
st.dataframe(pd.DataFrame(dataset_table()), hide_index=True, width="stretch")
st.markdown("##### Deliberately not predicted")
for k, v in UNSUPPORTED.items():
    st.markdown(f"- **{k.replace('_', ' ')}** — {v}")

st.markdown("##### Privacy")
st.caption("Scans are processed locally in memory/temporary storage, never uploaded to third-party APIs, and "
           "deleted after processing unless `app.keep_uploads` is enabled. DICOM identifiers are discarded; scans "
           "are referred to by a one-way pseudonymous ID. Do not upload identifiable data to shared deployments.")
nav_link("pages/1_Upload_CT.py", "Start: upload a CT scan", "⬆️")

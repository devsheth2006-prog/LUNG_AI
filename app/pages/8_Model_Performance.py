"""Model performance — every number here is read from files produced by the training/evaluation scripts."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import ACCENT, AMBER, MUTED, PLOT_LAYOUT, chip, page

from ctcancer.config import load_config
from ctcancer.inference.registry import models_dir
from ctcancer.inference.registry import all_statuses
from ctcancer.risk.tasks import TASKS

page("Model Performance")
st.markdown("Metrics are computed **once** on held-out, patient-level test sets by the training scripts and stored in "
            "model cards. Untrained models show *Model not trained* — no placeholder numbers exist in this app.")
FIG = load_config().path("figures_dir")
statuses = all_statuses()

for task, s in statuses.items():
    spec = TASKS[task]
    with st.expander(f"{spec.title}", expanded=s.state == "evaluated"):
        st.markdown(chip(s.state), unsafe_allow_html=True)
        st.caption(f"Reference label: {spec.label_kind}. {spec.caveat}")
        if s.state != "evaluated":
            st.caption(s.message)
            continue
        card, m = s.card, s.card["test_metrics"]
        st.caption(f"Model {card.get('architecture')} · datasets {', '.join(card.get('datasets', []))} · "
                   f"patients {card.get('n_patients', {})} · trained {card.get('created')}")
        keys = [k for k in ["roc_auc", "pr_auc", "recall_sensitivity", "specificity", "f1", "accuracy",
                            "balanced_accuracy", "brier", "ece", "dice", "iou", "sensitivity", "hd95_mm",
                            "macro_f1", "macro_roc_auc_ovr"] if isinstance(m.get(k), (int, float))]
        cols = st.columns(min(6, len(keys)) or 1)
        for i, k in enumerate(keys):
            lo = m.get(f"{k}_lo", (m.get(f"{k}_ci") or [None])[0])
            hi = m.get(f"{k}_hi", (m.get(f"{k}_ci") or [None, None])[1])
            cols[i % len(cols)].metric(k.replace("_", " "), f"{m[k]:.3f}",
                                       help=f"95% CI {lo:.3f}–{hi:.3f}" if lo is not None and hi is not None else None)
        if "confusion_matrix" in m:
            st.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["class_names"], columns=m["class_names"]))
        if card.get("history"):
            h = pd.DataFrame(card["history"])
            fig = go.Figure([go.Scatter(x=h.epoch, y=h.train_loss, name="train loss", line_color=ACCENT),
                             go.Scatter(x=h.epoch, y=h.val_loss, name="val loss", line_color=AMBER)])
            fig.update_layout(**PLOT_LAYOUT, height=280, title="Training / validation loss")
            st.plotly_chart(fig, width="stretch")
        if card.get("comparison"):
            st.markdown("**Model comparison**")
            st.dataframe(pd.DataFrame(card["comparison"]), hide_index=True, width="stretch")
        for png in sorted(FIG.glob(f"{task}_*.png")):
            st.image(str(png), caption=png.stem)

st.markdown("#### Representation study: radiomics vs CNN features, PCA, LDA")
f = models_dir() / "classical_study" / "results.json"
if f.exists():
    R = json.loads(f.read_text())
    df = pd.DataFrame(R["results"])
    fig = go.Figure(go.Bar(x=df.roc_auc, y=df.model, orientation="h", marker_color=ACCENT,
                           error_x=dict(type="data", symmetric=False, array=df.roc_auc_hi - df.roc_auc,
                                        arrayminus=df.roc_auc - df.roc_auc_lo, color=MUTED)))
    fig.update_layout(**PLOT_LAYOUT, height=40 + 26 * len(df), xaxis=dict(range=[0.4, 1], title="test ROC-AUC"),
                      yaxis=dict(autorange="reversed"), title=R["task"])
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(f"Development patients: {R['n_dev_patients']} (grouped 5-fold CV) · test patients: {R['n_test_patients']}. "
               "Configurations ranked by CV AUC; test AUC reported for all, with patient-bootstrap 95% CIs.")
else:
    st.markdown(chip("not_trained", "Study not run"), unsafe_allow_html=True)
    st.caption("Run `training/run_pca_lda.py` after extracting LIDC features.")

f = models_dir() / "multimodal_study" / "results.json"
st.markdown("#### Single-modality vs multimodal")
if f.exists():
    st.json(json.loads(f.read_text()))
else:
    st.markdown(chip("not_trained", "Study not run"), unsafe_allow_html=True)
    st.caption("Run `training/train_multimodal.py` (needs NSCLC features and the trained malignancy CNN).")

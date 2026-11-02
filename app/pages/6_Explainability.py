import json

import pandas as pd
import plotly.express as px
import streamlit as st

from ui import ACCENT, AMBER, PLOT_LAYOUT, page, require_scan

from ctcancer.classical.dimred import PCA_VS_LDA_TEXT
from ctcancer.config import load_config
from ctcancer.inference.registry import models_dir

page("Explainability")
res, A = require_scan()
st.info("This page explains the classical models: which radiomic features drive their decisions, and how benign and suspicious nodules are arranged in feature space (PCA and LDA).")

st.markdown("#### Radiomic feature attributions")
imp_png = load_config().path("figures_dir") / "radiomics_rf_importance.png"
if imp_png.exists():
    st.image(str(imp_png), caption="Random-forest impurity importance on the development set (run_pca_lda.py)")
else:
    st.caption("Not available until `training/run_pca_lda.py` has been run on real features. "
               "Optional SHAP values: `pip install shap`, see `ctcancer.explain.importance.shap_values`.")

st.markdown("#### PCA vs LDA")
st.markdown(PCA_VS_LDA_TEXT)
f = models_dir() / "classical_study" / "results.json"
if f.exists():
    R = json.loads(f.read_text())
    y = ["suspicious" if v else "likely benign" for v in R["test_labels"]]
    for name, P in R["projections"].items():
        l, r = st.columns(2)
        df = pd.DataFrame(P["pca2"], columns=["PC1", "PC2"]).assign(label=y)
        fig = px.scatter(df, x="PC1", y="PC2", color="label", color_discrete_sequence=[ACCENT, AMBER], opacity=0.7,
                         title=f"PCA · {name} ({P['pca_explained'][0]:.0%} + {P['pca_explained'][1]:.0%} variance)")
        fig.update_layout(**PLOT_LAYOUT, height=380)
        l.plotly_chart(fig, width="stretch")
        dl = pd.DataFrame({"LD1": [v[0] for v in P["lda"]], "label": y})
        fig = px.histogram(dl, x="LD1", color="label", barmode="overlay", nbins=40, opacity=0.65,
                           color_discrete_sequence=[ACCENT, AMBER], title=f"LDA · {name} (after PCA, 1 axis for 2 classes)")
        fig.update_layout(**PLOT_LAYOUT, height=380)
        r.plotly_chart(fig, width="stretch")
        st.caption(f"{name}: {P['pca_n_components_95']} principal components retain 95% of variance. "
                   "Projections fitted on development patients, plotted for held-out test patients.")
else:
    st.caption("Projections appear after running `training/extract_features.py --dataset lidc` and "
               "`training/run_pca_lda.py` on real LIDC data.")

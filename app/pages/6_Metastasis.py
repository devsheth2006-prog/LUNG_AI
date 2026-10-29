import streamlit as st

from ui import chip, disclaimer, gauge, page, require_scan

from ctcancer.risk.tasks import TASKS

page("Metastatic Spread Risk")
res, _ = require_scan()
disclaimer("Spread-risk estimates are derived from the primary tumour's CT appearance only; lymph nodes and "
           "distant organs are not themselves analysed.")
P = res.patient_predictions
cols = st.columns(3)
for col, key in zip(cols, ["n_involvement", "m_distant", "distant_recurrence"]):
    e = P.get(key, {})
    with col:
        st.markdown(f"**{TASKS[key].title}**<br>{chip(e.get('state', 'not_trained'))}", unsafe_allow_html=True)
        if e.get("probabilities"):
            pos = list(e["probabilities"].values())[1]
            st.plotly_chart(gauge(pos, "estimated probability"), width="stretch")
            st.markdown(f"Risk band: **{e['risk_band']}** <span class='muted'>(display cut-offs 33 % / 66 %, "
                        "not clinical thresholds)</span>", unsafe_allow_html=True)
            st.caption(e["wording"])
        else:
            st.caption(e.get("message", ""))
        st.caption(TASKS[key].caveat)

st.markdown("#### Anatomical sites")
st.markdown(f"{chip('unsupported')} **Metastasis prediction module — requires an appropriately labeled dataset.**",
            unsafe_allow_html=True)
st.markdown("Site-specific predictions (bone, liver, brain, adrenal glands) are **not** shown: none of the public "
            "datasets used provides per-organ metastasis labels, and fabricating them is out of the question. "
            "The software exposes a `TaskSpec` per site so a labelled cohort can be plugged in later "
            "(`src/ctcancer/risk/tasks.py`).")

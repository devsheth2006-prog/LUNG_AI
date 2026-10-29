import streamlit as st

from ui import chip, disclaimer, page, prob_bars, require_scan

from ctcancer.inference.registry import task_status

page("Staging / Risk")
res, _ = require_scan()
st.markdown("TNM concept: **T** = primary tumour characteristics, **N** = regional lymph-node involvement, "
            "**M** = distant metastasis. This page shows *model-estimated category probabilities* only — "
            "**model-estimated stage/risk category, requires clinical confirmation.** No overall stage "
            "(I–IV) is computed, because combining uncertain T/N/M estimates into an official stage would "
            "overstate what the models know.")
disclaimer()
P = res.patient_predictions
for key, letter in (("t_category", "T"), ("n_involvement", "N"), ("m_distant", "M")):
    e = P.get(key, {})
    st.markdown(f"#### {letter} — {e.get('task', key)}  {chip(e.get('state', 'not_trained'))}", unsafe_allow_html=True)
    if e.get("probabilities"):
        st.plotly_chart(prob_bars(e["probabilities"]), width="stretch")
        st.caption(f"{e['wording']} · {e.get('applies_to', '')}")
    else:
        st.caption(e.get("message", ""))
    st.caption(f"Caveat: {e.get('caveat', '')}")
    card = task_status(key).card
    if card and card.get("test_metrics"):
        m = card["test_metrics"]
        auc = m.get("roc_auc", m.get("macro_roc_auc_ovr"))
        st.caption(f"Held-out test: ROC-AUC {auc:.3f} (n={m.get('n_test_patients')} patients) · model "
                   f"{card.get('architecture')} · {card.get('staging_note', '')}")

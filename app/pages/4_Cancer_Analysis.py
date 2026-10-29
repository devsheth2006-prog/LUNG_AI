import pandas as pd
import streamlit as st

from ui import card, chip, disclaimer, gauge, page, pct, prob_bars, require_scan

page("Cancer Analysis")
res, A = require_scan()
P = res.patient_predictions
idx = next((l for l in res.lesions if l["id"] == res.index_lesion), None)

st.markdown("#### AI-assisted CT analysis")
c = st.columns(5)
with c[0]:
    card("Scan status", "✓ Processed", f"{res.runtime_s:.1f} s · {res.scan['format']}")
with c[1]:
    card("Suspicious lesion", "Candidate found" if idx else "None found",
         f"{len(res.lesions)} candidate(s)" + ("" if not idx or idx["detection"]["probability"] is not None
                                              else " · unscored"))
with c[2]:
    p = idx["malignancy"]["probability"] if idx else None
    card("Malignancy estimate", pct(p), "model-estimated" if p is not None else (idx["malignancy"]["status"] if idx else "—"))
with c[3]:
    v = idx["measurements"].get("volume_cm3") if idx else None
    card("Lesion volume", f"{v:.2f} cm³" if v else "—", idx["segmentation_method"][:40] if idx else "")
with c[4]:
    d = idx["measurements"].get("max_axial_diameter_mm") if idx else None
    card("Longest axial Ø", f"{d:.1f} mm" if d else "—", idx["location"]["text"][:44] if idx else "")
st.write("")
disclaimer()

if not res.lesions:
    st.info("No suspicious lesion candidates were detected. This does not exclude disease.")
    st.stop()

st.markdown("##### Lesion candidates")
rows = [{"ID": l["id"], "Location (approx.)": l["location"]["text"],
         "Axial Ø (mm)": l["measurements"].get("max_axial_diameter_mm"),
         "Volume (cm³)": l["measurements"].get("volume_cm3"), "Sphericity": l["measurements"].get("sphericity"),
         "Detection prob.": pct(l["detection"]["probability"]) if l["detection"]["probability"] is not None else "unscored",
         "Blobness (heuristic)": l["detection"]["blobness_index"],
         "Malignancy estimate": pct(l["malignancy"]["probability"]) if l["malignancy"]["probability"] is not None
         else l["malignancy"]["status"]} for l in res.lesions]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
if idx["detection"]["note"]:
    st.caption(idx["detection"]["note"])

l, r = st.columns(2)
with l:
    st.markdown("##### Malignancy (index lesion)")
    m = idx["malignancy"]
    if m["probability"] is None:
        st.markdown(chip("not_trained", m["status"]), unsafe_allow_html=True)
        st.caption("Model not trained — run `training/train_classifier.py --task malignancy`. No estimate is shown.")
    else:
        st.plotly_chart(prob_bars({"likely benign": m["benign_probability"], "suspicious": m["probability"]}),
                        width="stretch")
        st.caption(f"{m['wording']}. Model: {m['model']}, temperature-scaled (T={m['temperature']:.2f}).")
with r:
    st.markdown("##### Histological subtype")
    s = P.get("subtype", {})
    if s.get("probabilities"):
        st.plotly_chart(prob_bars({k.replace("_", " "): v for k, v in s["probabilities"].items()}),
                        width="stretch")
        st.caption(f"{s['wording']} · {s['applies_to']}")
    else:
        st.markdown(chip(s.get("state", "not_trained")), unsafe_allow_html=True)
        st.caption("Histological subtype prediction is unavailable for this scan/dataset. " + (s.get("message") or ""))
    st.caption(s.get("caveat", ""))

st.markdown("##### Spread-risk summary")
g = st.columns(3)
for col, key, title in zip(g, ["n_involvement", "m_distant", "distant_recurrence"],
                           ["Regional nodal involvement (N+)", "Distant metastasis (M1)", "Distant recurrence (ext.)"]):
    e = P.get(key, {})
    with col:
        if e.get("probabilities"):
            pos = list(e["probabilities"].values())[1]
            st.plotly_chart(gauge(pos, title), width="stretch")
            st.caption(f"{e.get('risk_band')} (display band) · {e['wording']}")
        else:
            st.markdown(f"**{title}**<br>{chip(e.get('state', 'not_trained'))}", unsafe_allow_html=True)
            st.caption(e.get("message", ""))

with st.expander("Radiomic features of the index lesion"):
    st.dataframe(pd.DataFrame(idx["radiomics"].items(), columns=["feature", "value"]), hide_index=True)

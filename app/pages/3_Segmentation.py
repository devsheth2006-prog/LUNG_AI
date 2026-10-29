import numpy as np
import streamlit as st

from ui import card, chip, ct_figure, page, require_scan

from ctcancer.inference.registry import task_status
from ctcancer.viz.images import representative_slice

page("Segmentation")
res, A = require_scan()

c = st.columns(3)
with c[0]:
    card("Lung volume", f"{res.lung['volume_ml']:.0f} mL" if res.lung.get("volume_ml") else "—",
         res.lung.get("method", ""))
with c[1]:
    card("Lung components", str(res.lung.get("components", "—")), "1 = right, 2 = left (LPS split)")
with c[2]:
    card("Lesion masks", str(len(res.lesions)), res.lesions[0]["segmentation_method"] if res.lesions else "none")

if A is not None:
    z = representative_slice(A, res.index_lesion)
    z = st.slider("Axial slice", 0, A.hu.shape[0] - 1, z)
    l, r = st.columns(2)
    with l:
        st.plotly_chart(ct_figure(A.hu[z], -600, 1500, title="Original CT (lung window)", height=440,
                                  is_hu=res.scan["hounsfield_units"]), width="stretch")
    with r:
        st.plotly_chart(ct_figure(A.hu[z], -600, 1500, A.lung_lr[z] > 0, A.lesion_labels[z] > 0,
                                  title="Lung mask (teal) + lesion contours (amber)", height=440,
                                  is_hu=res.scan["hounsfield_units"]), width="stretch")
    seg_only = np.where(A.lung_lr[z] > 0, A.hu[z], -1024)
    st.plotly_chart(ct_figure(seg_only, -600, 1500, title="Segmented lung region", height=380,
                              is_hu=res.scan["hounsfield_units"]), width="stretch")

st.markdown("##### Segmentation model evaluation (held-out patients)")
for task in ("lung_segmentation", "tumor_segmentation"):
    s = task_status(task)
    st.markdown(f"**{task.replace('_', ' ').title()}** {chip(s.state)}", unsafe_allow_html=True)
    if s.state == "evaluated":
        m = s.card["test_metrics"]
        cols = st.columns(5)
        for col, k in zip(cols, ["dice", "iou", "sensitivity", "specificity", "hd95_mm"]):
            ci = m.get(f"{k}_ci", [None, None])
            col.metric(k.upper() if k != "hd95_mm" else "HD95 (mm)", f"{m[k]:.3f}",
                       help=f"95% CI {ci[0]:.3f}–{ci[1]:.3f}" if ci[0] is not None else None)
        st.caption(f"Datasets: {', '.join(s.card.get('datasets', []))} · test patients: "
                   f"{m.get('n_test_patients')} · {s.card.get('metric_note', '')}")
    else:
        st.caption(s.message + (" The classical baseline in use has no validated accuracy." if task == "lung_segmentation" else ""))

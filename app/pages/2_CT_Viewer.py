import numpy as np
import plotly.graph_objects as go
import streamlit as st
from skimage import measure

from ui import PLOT_LAYOUT, WINDOWS, ct_figure, page, require_scan

page("CT Viewer")
res, A = require_scan()
if A is None:
    st.info("Image arrays are unavailable (API mode).")
    st.stop()
is_hu = bool(res.scan.get("hounsfield_units"))
sp = A.spacing

with st.sidebar:
    st.markdown("##### Display")
    plane = st.radio("Plane", ["Axial", "Coronal", "Sagittal"], horizontal=True)
    preset = st.selectbox("Window preset", list(WINDOWS), disabled=not is_hu)
    lv, wd = WINDOWS[preset]
    level = st.slider("Level (HU)", -1200, 1000, lv, disabled=not is_hu)
    width = st.slider("Width (HU)", 50, 2500, wd, disabled=not is_hu)
    show_lung = st.checkbox("Lung mask", True)
    show_les = st.checkbox("Lesion contours", True)
    show_cam = st.checkbox("Grad-CAM (index lesion slice)", False)

hu, lung, les = A.hu, A.lung_lr > 0, A.lesion_labels > 0
n = {"Axial": hu.shape[0], "Coronal": hu.shape[1], "Sagittal": hu.shape[2]}[plane]
default = n // 2
if res.index_lesion is not None and les.any():
    zz, yy, xx = np.where(A.lesion_labels == res.index_lesion)
    default = int({"Axial": np.median(zz), "Coronal": np.median(yy), "Sagittal": np.median(xx)}[plane])
idx = st.slider(f"{plane} slice", 0, n - 1, default)

if plane == "Axial":
    img, lm, sm, aspect = hu[idx], lung[idx], les[idx], sp[1] / sp[2]
elif plane == "Coronal":
    img, lm, sm, aspect = hu[::-1, idx, :], lung[::-1, idx, :], les[::-1, idx, :], sp[0] / sp[2]
else:
    img, lm, sm, aspect = hu[::-1, :, idx], lung[::-1, :, idx], les[::-1, :, idx], sp[0] / sp[1]

cam = None
if show_cam and plane == "Axial" and res.index_lesion in A.cams and A.cams[res.index_lesion]["z"] == idx:
    c = A.cams[res.index_lesion]
    cam = np.full(img.shape, np.nan)
    s, (cy, cx) = c["size"], c["centre_yx"]
    y0, x0 = int(round(cy)) - s // 2, int(round(cx)) - s // 2
    ys, xs = slice(max(y0, 0), min(y0 + s, img.shape[0])), slice(max(x0, 0), min(x0 + s, img.shape[1]))
    cam[ys, xs] = c["gradcam"][ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
elif show_cam:
    st.caption("Grad-CAM is available on the axial slice through the index lesion centre when the malignancy model is trained.")

st.plotly_chart(ct_figure(img, level, width, lm if show_lung else None, sm if show_les else None, cam, aspect,
                          f"{plane} {idx + 1}/{n}", is_hu), width="stretch",
                config={"scrollZoom": True, "displaylogo": False})
st.caption("Scroll to zoom, drag to pan, double-click to reset. Orientation: radiological convention "
           "(patient right on screen left). Voxels shown at the resampled spacing "
           f"{tuple(round(s, 2) for s in sp)} mm (z, y, x).")

with st.expander("3-D rendering (lungs + lesions)"):
    step = st.select_slider("Downsampling (speed vs detail)", [2, 3, 4, 6], value=4)
    if st.button("Render 3-D"):
        fig = go.Figure()
        for mask, color, op, name in ((lung, "#4fb3bf", 0.15, "lungs"), (les, "#f2b134", 0.9, "lesions")):
            m = mask[::step, ::step, ::step]
            if m.sum() < 10:
                continue
            v, f, _, _ = measure.marching_cubes(m.astype(np.uint8), 0.5, spacing=tuple(s * step for s in sp))
            fig.add_trace(go.Mesh3d(x=v[:, 2], y=v[:, 1], z=v[:, 0], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                    color=color, opacity=op, name=name))
        fig.update_layout(**PLOT_LAYOUT, height=560, scene=dict(aspectmode="data", xaxis_visible=False,
                                                                yaxis_visible=False, zaxis_visible=False))
        st.plotly_chart(fig, width="stretch")

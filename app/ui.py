"""Shared Streamlit UI: theme, layout primitives, status chips, gauges and the CT viewer."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from ctcancer import DISCLAIMER  # noqa: E402

INK, MUTED, PANEL, LINE = "#e6edf3", "#8b98a5", "#131a22", "#253140"
ACCENT, AMBER, RED, GREEN = "#4fb3bf", "#f2b134", "#e5534b", "#57ab5a"
PLOT_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=12),
                   margin=dict(l=10, r=10, t=36, b=10))

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'IBM Plex Sans', sans-serif; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 5rem !important; max-width: 1280px; }}
[data-testid="stHeader"] {{ background: rgba(14,17,23,0.96); }}
h1, h2, h3 {{ letter-spacing: -0.01em; font-weight: 600; }}
.ct-header {{ display:flex; align-items:baseline; justify-content:space-between; border-bottom:1px solid {LINE};
  padding-bottom:.6rem; margin-bottom:1rem; }}
.ct-header .t {{ font-size:1.35rem; font-weight:600; color:{INK}; }}
.ct-header .s {{ font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.08em; }}
.ct-card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:10px; padding:14px 16px; height:100%; }}
.ct-card .k {{ font-size:.72rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.07em; }}
.ct-card .v {{ font-family:'IBM Plex Mono',monospace; font-size:1.55rem; color:{INK}; margin-top:2px; }}
.ct-card .n {{ font-size:.78rem; color:{MUTED}; margin-top:4px; }}
.chip {{ display:inline-block; font-size:.7rem; padding:2px 8px; border-radius:999px; border:1px solid; font-weight:500;
  letter-spacing:.02em; white-space:nowrap; }}
.chip.ok {{ color:{GREEN}; border-color:{GREEN}55; background:{GREEN}14; }}
.chip.warn {{ color:{AMBER}; border-color:{AMBER}55; background:{AMBER}14; }}
.chip.off {{ color:{MUTED}; border-color:{LINE}; background:#ffffff08; }}
.chip.bad {{ color:{RED}; border-color:{RED}55; background:{RED}14; }}
.disclaimer {{ border-left:3px solid {AMBER}; background:{AMBER}12; padding:10px 14px; border-radius:6px; font-size:.85rem; color:{INK}; }}
.muted {{ color:{MUTED}; font-size:.85rem; }}
.mono {{ font-family:'IBM Plex Mono',monospace; }}
.step {{ display:flex; gap:10px; align-items:center; padding:5px 0; border-bottom:1px dashed {LINE}; font-size:.9rem; }}
.step .i {{ font-family:'IBM Plex Mono',monospace; color:{MUTED}; width:26px; }}
section[data-testid="stSidebar"] {{ border-right:1px solid {LINE}; }}
div[data-testid="stMetricValue"] {{ font-family:'IBM Plex Mono',monospace; }}
</style>
"""


def page(title: str, subtitle: str = "Research prototype Â· not for clinical use") -> None:
    st.set_page_config(page_title=f"{title} Â· CT Lung Research", page_icon="ðŸ«", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="ct-header"><span class="t">{title}</span><span class="s">{subtitle}</span></div>',
                unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**AI-Assisted CT Lung Research Platform**")
        st.caption("Data are processed locally in this session and discarded when it ends. "
                   "Nothing is sent to external services.")
        r = st.session_state.get("result")
        if r is not None and r.ok:
            st.markdown(f'<span class="chip ok">Scan loaded</span> <span class="mono muted">{r.scan["scan_id"]}</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="chip off">No scan loaded</span>', unsafe_allow_html=True)


def disclaimer(extra: str = "") -> None:
    st.markdown(f'<div class="disclaimer"><b>Model-generated research estimates â€” not a medical diagnosis.</b> '
                f'{DISCLAIMER} {extra}</div>', unsafe_allow_html=True)


def chip(state: str, text: str | None = None) -> str:
    cls = {"evaluated": "ok", "trained_unevaluated": "warn", "not_trained": "off", "refused": "warn",
           "unsupported": "bad"}.get(state, "off")
    label = text or {"evaluated": "Trained & evaluated", "trained_unevaluated": "Not yet evaluated",
                     "not_trained": "Model not trained", "refused": "Insufficient labels",
                     "unsupported": "Not supported by data"}.get(state, state)
    return f'<span class="chip {cls}">{label}</span>'


def card(key: str, value: str, note: str = "") -> None:
    st.markdown(f'<div class="ct-card"><div class="k">{key}</div><div class="v">{value}</div>'
                f'<div class="n">{note}</div></div>', unsafe_allow_html=True)


def require_scan():
    r = st.session_state.get("result")
    if r is None:
        st.info("No scan analysed yet. Go to **Upload CT** first.")
        nav_link("pages/1_Upload_CT.py", "Open Upload CT", "â¬†ï¸")
        st.stop()
    if not r.ok:
        st.error(r.error)
        st.stop()
    return r, st.session_state.get("arrays")


def nav_link(path: str, label: str, icon: str | None = None, container=None) -> None:
    """st.page_link that degrades to plain text when the page registry is unavailable (e.g. tests)."""
    try:
        (container or st).page_link(path, label=label, icon=icon)
    except Exception:
        (container or st).caption(label)


def pct(p) -> str:
    return "â€”" if p is None else f"{100 * p:.0f}%"


def gauge(p: float | None, title: str, bands=(0.33, 0.66)) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=None if p is None else 100 * p, number={"suffix": "%", "font": {"size": 30}},
        title={"text": title, "font": {"size": 13, "color": MUTED}},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED}, "bar": {"color": INK, "thickness": 0.25},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 100 * bands[0]], "color": "#1f3a2a"},
                         {"range": [100 * bands[0], 100 * bands[1]], "color": "#3d3420"},
                         {"range": [100 * bands[1], 100], "color": "#43221f"}]}))
    fig.update_layout(**PLOT_LAYOUT, height=210)
    return fig


def prob_bars(probs: dict[str, float], title: str = "") -> go.Figure:
    names, vals = list(probs), [100 * v for v in probs.values()]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=ACCENT,
                           text=[f"{v:.0f}%" for v in vals], textposition="outside"))
    fig.update_layout(**PLOT_LAYOUT, height=60 + 38 * len(names), title=title,
                      xaxis=dict(range=[0, 110], showgrid=False, visible=False), yaxis=dict(autorange="reversed"))
    return fig


# ------------------------------------------------------------------ CT viewer
WINDOWS = {"Lung": (-600, 1500), "Mediastinum": (40, 400), "Bone": (400, 1800), "Soft tissue": (50, 350)}


def ct_figure(img_hu: np.ndarray, level: float, width: float, lung=None, lesion=None, cam=None,
              aspect: float = 1.0, title: str = "", is_hu: bool = True, height: int = 560) -> go.Figure:
    lo, hi = level - width / 2, level + width / 2
    if not is_hu:
        lo, hi = float(np.percentile(img_hu, 0.5)), float(np.percentile(img_hu, 99.5))
    fig = go.Figure(go.Heatmap(z=img_hu, colorscale="gray", zmin=lo, zmax=hi, showscale=False,
                               hovertemplate="x %{x} Â· y %{y}<br>%{z:.0f}" + (" HU" if is_hu else "") + "<extra></extra>"))
    if lung is not None and lung.any():
        fig.add_trace(go.Heatmap(z=np.where(lung, 1.0, np.nan), colorscale=[[0, ACCENT], [1, ACCENT]],
                                 opacity=0.22, showscale=False, hoverinfo="skip"))
    if cam is not None:
        fig.add_trace(go.Heatmap(z=np.where(cam > 0.15, cam, np.nan), colorscale="Inferno", opacity=0.55,
                                 zmin=0, zmax=1, showscale=False, hoverinfo="skip"))
    if lesion is not None and lesion.any():
        fig.add_trace(go.Contour(z=lesion.astype(float), contours=dict(start=0.5, end=0.5, coloring="lines"),
                                 line=dict(color=AMBER, width=2), showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=aspect, visible=False)
    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_layout(**PLOT_LAYOUT, height=height, title=title, dragmode="pan")
    return fig


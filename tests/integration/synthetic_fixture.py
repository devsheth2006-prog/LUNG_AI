"""SOFTWARE-TEST FIXTURE — synthetic geometric patches in the cached-data layout.

Purpose: prove that every training/evaluation script runs end-to-end (shapes, file formats,
splits, model cards, app integration) without downloading real data. The "labels" are
geometric (smooth sphere vs lobulated blob) and the clinical columns are random; nothing
here resembles patient data and the resulting metrics are meaningless. The fixture is written
to a temporary directory and never into data/ or models/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _blob(rng, size, radius, lobulated: bool):
    z, y, x = np.mgrid[:size, :size, :size] - size // 2
    r = np.sqrt(z ** 2 + y ** 2 + x ** 2) + 1e-6
    if lobulated:
        theta, phi = np.arccos(z / r), np.arctan2(y, x)
        radius = radius * (1 + 0.35 * np.sin(4 * phi) * np.sin(3 * theta))
    mask = r <= radius
    hu = rng.normal(-850, 30, (size,) * 3)
    hu[mask] = rng.normal(30, 25, mask.sum())
    return hu.astype(np.float16), mask


def build(root: Path, n_patients: int = 40, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    proc, splits = root / "processed", root / "splits"
    rows = []
    (proc / "lidc" / "patches").mkdir(parents=True, exist_ok=True)
    for p in range(n_patients):
        for k in range(2):
            lab = int(rng.random() < 0.5)
            hu, m = _blob(rng, 64, rng.uniform(4, 10), bool(lab))
            lid = f"SYN_{p}_{k}"
            np.savez_compressed(proc / "lidc" / "patches" / f"{lid}.npz", hu=hu, mask=m, spacing=np.ones(3, np.float32))
            rows.append({"lesion_id": lid, "patient_id": f"SYNP{p:03d}", "label": lab, "patch_path": f"patches/{lid}.npz"})
    lidc = pd.DataFrame(rows)
    lidc.to_csv(proc / "lidc" / "lesions.csv", index=False)

    (proc / "luna16_candidates" / "cand").mkdir(parents=True, exist_ok=True)
    crows = []
    for i, r in lidc.iterrows():
        d = np.load(proc / "lidc" / r.patch_path)
        hu = d["hu"][8:56, 8:56, 8:56] if i % 2 == 0 else rng.normal(-850, 30, (48,) * 3).astype(np.float16)
        np.savez_compressed(proc / "luna16_candidates" / "cand" / f"c{i}.npz", hu=hu)
        crows.append({"lesion_id": f"c{i}", "patient_id": r.patient_id, "label": int(i % 2 == 0),
                      "patch_path": f"cand/c{i}.npz"})
    pd.DataFrame(crows).to_csv(proc / "luna16_candidates" / "candidates.csv", index=False)

    (proc / "luna16_lung" / "slices").mkdir(parents=True, exist_ok=True)
    srows = []
    yy, xx = np.mgrid[:96, :96]
    for p in range(12):
        for z in range(4):
            m = (((yy - 48) / 30) ** 2 + ((xx - 28) / 14) ** 2 <= 1) | (((yy - 48) / 30) ** 2 + ((xx - 68) / 14) ** 2 <= 1)
            img = np.where(m, 0.15, 0.6) + rng.normal(0, 0.03, m.shape)
            np.savez_compressed(proc / "luna16_lung" / "slices" / f"s{p}_{z}.npz", img=img.astype(np.float16), mask=m)
            srows.append({"slice_id": f"s{p}_{z}", "patient_id": f"L{p}", "path": f"slices/s{p}_{z}.npz"})
    pd.DataFrame(srows).to_csv(proc / "luna16_lung" / "lung_slices.csv", index=False)

    (proc / "nsclc_radiomics" / "patches").mkdir(parents=True, exist_ok=True)
    nrows = []
    hist = ["adenocarcinoma", "squamous_cell_carcinoma", "other_nsclc"]
    for p in range(60):
        npos = int(rng.random() < 0.5)
        hu, m = _blob(rng, 96, rng.uniform(6, 14) + 4 * npos, bool(npos))
        lid = f"NS_{p}"
        np.savez_compressed(proc / "nsclc_radiomics" / "patches" / f"{lid}.npz", hu=hu, mask=m, spacing=np.ones(3, np.float32))
        nrows.append({"lesion_id": lid, "patient_id": f"NSP{p:03d}", "dataset": "nsclc_radiomics",
                      "patch_path": f"patches/{lid}.npz", "age": float(rng.integers(45, 85)),
                      "sex_male": float(rng.integers(0, 2)), "histology": hist[p % 3],
                      "t_category": float(1 + p % 4), "n_category": float(npos * 2), "n_positive": float(npos),
                      "m_positive": float(p % 30 == 0)})
    pd.DataFrame(nrows).to_csv(proc / "nsclc_radiomics" / "lesions.csv", index=False)

    from ctcancer.data.splits import patient_level_split, save_split

    for name, df in (("lidc", lidc), ("luna16_candidates", pd.DataFrame(crows)),
                     ("luna16_lung", pd.DataFrame(srows)), ("nsclc_radiomics", pd.DataFrame(nrows))):
        lab = "label" if "label" in df else None
        save_split(patient_level_split(df, "patient_id", lab, (0.7, 0.15, 0.15), seed), splits / f"{name}.json")
    (root / "FIXTURE_README.json").write_text(json.dumps({"synthetic": True, "purpose": "software test only"}))
    return root

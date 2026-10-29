"""Parsers that turn the official clinical CSVs into harmonised label tables.

Column names differ between releases, so columns are located by case/space-insensitive
aliases. If a required column is not found, a clear error lists what was available —
nothing is ever filled in or guessed.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_col(df: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
    table = {_norm(c): c for c in df.columns}
    for a in aliases:
        if _norm(a) in table:
            return table[_norm(a)]
    if required:
        raise KeyError(f"None of {aliases} found. Available columns: {list(df.columns)}")
    return None


def harmonise_histology(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in {"", "na", "nan", "n/a", "unknown"}:
        return None
    if "adeno" in s:
        return "adenocarcinoma"
    if "squam" in s:
        return "squamous_cell_carcinoma"
    if "small cell" in s and "non" not in s:
        return "small_cell"  # will be counted; never present in these cohorts as far as we know
    if "large" in s or "nos" in s or "nsclc" in s or "other" in s:
        return "other_nsclc"
    return "other_nsclc"


def _stage_int(v, prefix: str) -> int | None:
    """'T2a' -> 2, 'N0' -> 0, 2 -> 2, 'Tis' -> 0, 'TX'/'NA' -> None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().upper().replace(prefix, "")
    if s.startswith("IS"):
        return 0
    m = re.match(r"^(\d)", s)
    return int(m.group(1)) if m else None


def parse_nsclc_radiomics(csv_path: str | Path) -> pd.DataFrame:
    """NSCLC-Radiomics (Lung1) clinical CSV -> harmonised labels (CLINICAL TNM)."""
    df = pd.read_csv(csv_path)
    pid = find_col(df, ["PatientID", "Patient ID"])
    out = pd.DataFrame({"patient_id": df[pid].astype(str)})
    out["age"] = pd.to_numeric(df[find_col(df, ["age"])], errors="coerce")
    out["sex_male"] = df[find_col(df, ["gender", "sex"])].astype(str).str.lower().map(
        {"male": 1, "female": 0, "m": 1, "f": 0})
    out["histology"] = df[find_col(df, ["Histology"])].map(harmonise_histology)
    t = df[find_col(df, ["clinical.T.Stage", "T stage"])].map(lambda v: _stage_int(v, "T"))
    n = df[find_col(df, ["Clinical.N.Stage", "N stage"])].map(lambda v: _stage_int(v, "N"))
    m = df[find_col(df, ["Clinical.M.Stage", "M stage"])].map(lambda v: _stage_int(v, "M"))
    # Lung1 encodes unknown T as 5 in some releases -> treat values outside 1..4 as missing
    out["t_category"] = t.where(t.between(1, 4))
    out["n_category"] = n.where(n.between(0, 3))
    out["n_positive"] = (out["n_category"] > 0).astype("float").where(out["n_category"].notna())
    out["m_positive"] = m.where(m.isin([0, 1])).astype("float")
    out["staging_type"] = "clinical"
    out["source"] = "NSCLC-Radiomics"
    return out


def parse_nsclc_radiogenomics(csv_path: str | Path) -> pd.DataFrame:
    """NSCLC-Radiogenomics clinical CSV -> harmonised labels (PATHOLOGICAL TNM)."""
    df = pd.read_csv(csv_path)
    out = pd.DataFrame({"patient_id": df[find_col(df, ["Case ID", "CaseID"])].astype(str)})
    out["age"] = pd.to_numeric(df[find_col(df, ["Age at Histological Diagnosis", "Age"])], errors="coerce")
    out["sex_male"] = df[find_col(df, ["Gender", "Sex"])].astype(str).str.lower().map({"male": 1, "female": 0})
    smk = find_col(df, ["Smoking status"], required=False)
    if smk:
        s = df[smk].astype(str).str.lower()
        out["smoker_current"] = (s == "current").astype(float).where(s.isin(["current", "former", "nonsmoker"]))
        out["smoker_former"] = (s == "former").astype(float).where(s.isin(["current", "former", "nonsmoker"]))
    pk = find_col(df, ["Pack Years"], required=False)
    if pk:
        out["pack_years"] = pd.to_numeric(df[pk], errors="coerce")
    out["histology"] = df[find_col(df, ["Histology", "Histology "])].map(harmonise_histology)
    t = df[find_col(df, ["Pathological T stage"])].map(lambda v: _stage_int(v, "T"))
    n = df[find_col(df, ["Pathological N stage"])].map(lambda v: _stage_int(v, "N"))
    m = df[find_col(df, ["Pathological M stage"])].map(lambda v: _stage_int(v, "M"))
    out["t_category"] = t.where(t.between(0, 4))
    out["n_category"] = n.where(n.between(0, 3))
    out["n_positive"] = (out["n_category"] > 0).astype("float").where(out["n_category"].notna())
    out["m_positive"] = m.where(m.isin([0, 1])).astype("float")
    rl = find_col(df, ["Recurrence Location"], required=False)
    if rl:
        loc = df[rl].astype(str).str.lower()
        out["distant_recurrence"] = loc.str.contains("distant").astype(float)
        out.loc[df[rl].isna(), "distant_recurrence"] = np.nan
    out["staging_type"] = "pathological"
    out["source"] = "NSCLC-Radiogenomics"
    return out


CLINICAL_FEATURES = ["age", "sex_male", "smoker_current", "smoker_former", "pack_years"]


def fit_clinical_imputer(train_df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, float]:
    """Median fill values computed on the TRAINING patients only (avoids test-set leakage)."""
    cols = [c for c in (columns or CLINICAL_FEATURES) if c in train_df and train_df[c].notna().any()]
    return {c: float(np.nanmedian(train_df[c].astype(float))) for c in cols}


def clinical_matrix(df: pd.DataFrame, fill_values: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """Numeric clinical matrix with an explicit missing-indicator column per variable."""
    feats, names = [], []
    for c, fill in fill_values.items():
        v = df[c].astype(float).to_numpy() if c in df else np.full(len(df), np.nan)
        miss = np.isnan(v)
        feats += [np.where(miss, fill, v), miss.astype(float)]
        names += [c, f"{c}__missing"]
    return (np.stack(feats, 1) if feats else np.zeros((len(df), 0))), names

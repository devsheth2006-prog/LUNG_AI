"""Patient-level data splitting — the single most important guard against optimistic results.

Lesions, slices and patches from one patient are strongly correlated (same scanner, same
anatomy, often the same nodule seen by 4 readers in LIDC). Splitting at slice or lesion level
lets a model "recognise the patient" and inflates every metric. Every split in this project is
made on unique patient IDs, stratified on a per-patient label, and then asserted leak-free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def patient_labels(df: pd.DataFrame, patient_col: str, label_col: str | None) -> pd.Series:
    """One stratification label per patient (max label = 'patient has any positive lesion')."""
    if label_col is None or label_col not in df:
        return pd.Series(0, index=df[patient_col].unique())
    return df.groupby(patient_col)[label_col].max()


def patient_level_split(df: pd.DataFrame, patient_col: str = "patient_id", label_col: str | None = "label",
                        fractions=(0.70, 0.15, 0.15), seed: int = 42) -> dict[str, list[str]]:
    assert abs(sum(fractions) - 1) < 1e-6, "split fractions must sum to 1"
    labels = patient_labels(df, patient_col, label_col)
    pids = labels.index.to_numpy()
    y = labels.to_numpy()
    strat = y if _can_stratify(y, fractions) else None
    train, rest = train_test_split(pids, test_size=1 - fractions[0], random_state=seed, stratify=strat)
    rest_y = labels.loc[rest].to_numpy()
    rel_test = fractions[2] / (fractions[1] + fractions[2])
    strat2 = rest_y if _can_stratify(rest_y, (1 - rel_test, rel_test)) else None
    val, test = train_test_split(rest, test_size=rel_test, random_state=seed, stratify=strat2)
    split = {"train": sorted(map(str, train)), "val": sorted(map(str, val)), "test": sorted(map(str, test))}
    assert_no_leakage(split)
    return split


def patient_holdout(df: pd.DataFrame, test_frac: float = 0.15, patient_col: str = "patient_id",
                    label_col: str | None = "label", seed: int = 42) -> dict[str, list[str]]:
    """Two-way patient-level split (development / test) for cross-validated small-cohort tasks."""
    labels = patient_labels(df, patient_col, label_col)
    pids, y = labels.index.to_numpy(), labels.to_numpy()
    strat = y if _can_stratify(y, (1 - test_frac, test_frac)) else None
    dev, test = train_test_split(pids, test_size=test_frac, random_state=seed, stratify=strat)
    split = {"dev": sorted(map(str, dev)), "test": sorted(map(str, test))}
    assert_no_leakage(split)
    return split


def _can_stratify(y: np.ndarray, fractions) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and counts.min() >= max(2, int(np.ceil(1 / min(fractions))))


def patient_group_kfold(df: pd.DataFrame, n_splits: int = 5, patient_col: str = "patient_id",
                        label_col: str = "label", seed: int = 42):
    """Stratified *grouped* K-fold for small datasets (risk models on ~200 patients)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from sgkf.split(df, df[label_col], groups=df[patient_col])


def assert_no_leakage(split: dict[str, list[str]]) -> None:
    names = list(split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(split[a]) & set(split[b])
            if overlap:
                raise AssertionError(f"Patient leakage between {a} and {b}: {sorted(overlap)[:5]}")


def save_split(split: dict, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, "counts": {k: len(v) for k, v in split.items()}, **(meta or {})}
    path.write_text(json.dumps(payload, indent=2))


def load_split(path: str | Path) -> dict[str, list[str]]:
    split = json.loads(Path(path).read_text())["split"]
    assert_no_leakage(split)
    return split

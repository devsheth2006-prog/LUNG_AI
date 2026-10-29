import numpy as np
import pandas as pd
import pytest

from ctcancer.data.clinical import clinical_matrix, fit_clinical_imputer, harmonise_histology, parse_nsclc_radiomics
from ctcancer.data.splits import assert_no_leakage, patient_holdout, patient_level_split
from ctcancer.evaluation.metrics import binary_metrics, bootstrap_ci, expected_calibration_error


def _lesions(n_pat=100, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        lab = int(rng.random() < 0.3)
        for k in range(rng.integers(1, 5)):  # several lesions per patient
            rows.append({"patient_id": f"P{p}", "lesion": k, "label": lab})
    return pd.DataFrame(rows)


def test_patient_level_split_has_no_leakage_and_is_stratified():
    df = _lesions()
    s = patient_level_split(df, fractions=(0.7, 0.15, 0.15), seed=1)
    assert_no_leakage(s)
    assert sum(len(v) for v in s.values()) == df.patient_id.nunique()
    prev = {k: df[df.patient_id.isin(v)].groupby("patient_id").label.max().mean() for k, v in s.items()}
    assert max(prev.values()) - min(prev.values()) < 0.15
    assert abs(len(s["train"]) / 100 - 0.7) < 0.03


def test_split_is_reproducible():
    df = _lesions()
    assert patient_level_split(df, seed=3) == patient_level_split(df, seed=3)


def test_leakage_detected():
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage({"train": ["a", "b"], "test": ["b"]})


def test_holdout():
    s = patient_holdout(_lesions(), 0.2)
    assert_no_leakage(s)


def test_clinical_imputer_uses_train_only():
    tr = pd.DataFrame({"age": [50.0, 60.0, np.nan]})
    te = pd.DataFrame({"age": [np.nan, 99.0]})
    fills = fit_clinical_imputer(tr, ["age"])
    X, names = clinical_matrix(te, fills)
    assert names == ["age", "age__missing"]
    assert X[0, 0] == 55.0 and X[0, 1] == 1.0 and X[1, 0] == 99.0


def test_histology_and_lung1_parsing(tmp_path):
    assert harmonise_histology("Adenocarcinoma") == "adenocarcinoma"
    assert harmonise_histology("squamous cell carcinoma") == "squamous_cell_carcinoma"
    assert harmonise_histology("large cell") == "other_nsclc"
    assert harmonise_histology(np.nan) is None
    csv = tmp_path / "lung1.csv"
    pd.DataFrame({"PatientID": ["A", "B", "C"], "age": [70, 61, 55], "clinical.T.Stage": [2, 5, 4],
                  "Clinical.N.Stage": [0, 2, 3], "Clinical.M.Stage": [0, 0, 3], "Overall.Stage": ["I", "IIIa", "IIIb"],
                  "Histology": ["adenocarcinoma", "nos", None], "gender": ["male", "female", "male"],
                  "Survival.time": [100, 200, 300], "deadstatus.event": [1, 0, 1]}).to_csv(csv, index=False)
    df = parse_nsclc_radiomics(csv)
    assert list(df.t_category.isna()) == [False, True, False]      # T=5 (unknown) -> missing, not guessed
    assert list(df.n_positive) == [0.0, 1.0, 1.0]
    assert np.isnan(df.m_positive[2])                              # M=3 (Mx) -> missing
    assert pd.isna(df.histology[2])


def test_binary_metrics_and_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    m = binary_metrics(y, p)
    assert 0.8 < m["roc_auc"] <= 1 and m["tp"] + m["fn"] == y.sum()
    from sklearn.metrics import roc_auc_score

    est, lo, hi = bootstrap_ci(roc_auc_score, y, p, groups=np.arange(400) // 2, n_boot=200)
    assert lo <= est <= hi
    assert expected_calibration_error(y, y.astype(float)) == pytest.approx(0)

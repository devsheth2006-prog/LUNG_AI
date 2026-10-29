"""Generic trainer for patient-level tabular tasks (subtype, T, N, M, recurrence).

Protocol (small cohorts, so no single fixed validation set):
  1. Patient-level stratified hold-out test set (15 %), never touched until the end.
  2. On the remaining patients: stratified *grouped* 5-fold CV compares every
     (feature set × reduction × classifier) combination.
  3. The best configuration by mean CV ROC-AUC is refitted on all development patients,
     calibrated (isotonic/sigmoid via CalibratedClassifierCV), and evaluated ONCE on the test set
     with patient-bootstrap 95 % CIs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ..classical.dimred import ablation_configs, build_pipeline
from ..data.splits import patient_holdout
from ..evaluation.metrics import binary_metrics, bootstrap_ci, multiclass_metrics, youden_threshold
from ..logging_utils import get_logger
from .tasks import TaskSpec

log = get_logger(__name__)


class InsufficientLabelsError(RuntimeError):
    pass


@dataclass
class TabularResult:
    task: str
    cv_table: pd.DataFrame
    best_name: str
    test_metrics: dict
    model_path: Path


def _encode(y: pd.Series, spec: TaskSpec) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(y):
        mapping = {c: i for i, c in enumerate(spec.classes)}
        return y.map(mapping).to_numpy()
    if spec.key == "t_category":
        return (y.astype(int) - 1).clip(0, 3).to_numpy()   # T0/Tis merged into T1
    return y.astype(int).to_numpy()


def _score(y, prob, n_classes):
    if n_classes == 2:
        return roc_auc_score(y, prob[:, 1])
    return roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=list(range(n_classes)))


def train_tabular_task(df: pd.DataFrame, spec: TaskSpec, radiomic_cols, deep_cols, clinical_cols,
                       out_dir: str | Path, cfg, seed: int = 42) -> TabularResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = df[df[spec.label_col].notna()].copy()
    y_all = _encode(data[spec.label_col], spec)
    keep = ~pd.isna(y_all)
    data, y_all = data[keep], y_all[keep].astype(int)
    counts = np.bincount(y_all, minlength=len(spec.classes))
    min_pos = cfg.get_path("risk.min_positive_labels", 15)
    if spec.key == "t_category" and counts[3] < min_pos and counts[2] + counts[3] >= min_pos:
        log.warning("Merging T4 into T3 (T4 n=%d < %d): classes become T1, T2, T3–T4", counts[3], min_pos)
        y_all = np.minimum(y_all, 2)
        spec = replace(spec, classes=("T1", "T2", "T3–T4"))
        counts = np.bincount(y_all, minlength=len(spec.classes))
    if (counts < min_pos).any():
        rare = [f"{c}: n={n}" for c, n in zip(spec.classes, counts) if n < min_pos]
        raise InsufficientLabelsError(
            f"Task '{spec.key}' refused: classes below {min_pos} labelled patients ({', '.join(rare)}). "
            "A model trained on so few positives would produce unreliable probabilities.")
    data = data.assign(_y=y_all)
    split = patient_holdout(data, 0.15, "patient_id", "_y", seed)
    dev = data[data.patient_id.isin(split["dev"])]
    test = data[data.patient_id.isin(split["test"])]
    n_classes = len(spec.classes)
    cv = StratifiedGroupKFold(n_splits=cfg.get_path("classical.cv_folds", 5), shuffle=True, random_state=seed)
    rows = []
    for name, cols, red in ablation_configs(radiomic_cols, deep_cols, clinical_cols):
        for clf in cfg.get_path("risk.models", ["logistic", "random_forest", "gradient_boosting"]):
            if red in ("lda", "pca_lda") and clf != "logistic":
                continue
            scores = []
            for tr, va in cv.split(dev, dev._y, groups=dev.patient_id):
                pipe = build_pipeline(red, clf, cfg.get_path("classical.pca_variance", 0.95), seed)
                pipe.fit(dev.iloc[tr][cols].to_numpy(float), dev._y.iloc[tr])
                prob = pipe.predict_proba(dev.iloc[va][cols].to_numpy(float))
                try:
                    scores.append(_score(dev._y.iloc[va].to_numpy(), prob, n_classes))
                except ValueError:
                    pass
            rows.append({"features": name, "reduction": red, "classifier": clf, "n_features": len(cols),
                         "cv_auc_mean": float(np.mean(scores)) if scores else np.nan,
                         "cv_auc_std": float(np.std(scores)) if scores else np.nan, "_cols": cols})
    table = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
    best = table.iloc[0]
    log.info("%s best CV config: %s / %s / %s (AUC %.3f)", spec.key, best.features, best.reduction,
             best.classifier, best.cv_auc_mean)
    cols = list(best._cols)
    base = build_pipeline(best.reduction, best.classifier, cfg.get_path("classical.pca_variance", 0.95), seed)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(dev[cols].to_numpy(float), dev._y)
    prob = model.predict_proba(test[cols].to_numpy(float))
    yt = test._y.to_numpy()
    if n_classes == 2:
        thr = youden_threshold(dev._y, model.predict_proba(dev[cols].to_numpy(float))[:, 1])
        m = binary_metrics(yt, prob[:, 1], thr)
        _, lo, hi = bootstrap_ci(roc_auc_score, yt, prob[:, 1], test.patient_id.to_numpy(),
                                 cfg.get_path("evaluation.bootstrap_iterations", 1000))
        m["roc_auc_ci"] = [lo, hi]
    else:
        m = multiclass_metrics(yt, prob, list(spec.classes))
    m.update({"n_dev_patients": int(dev.patient_id.nunique()), "n_test_patients": int(test.patient_id.nunique()),
              "class_counts": dict(zip(spec.classes, map(int, counts)))})
    path = out_dir / "model.joblib"
    joblib.dump({"model": model, "columns": cols, "classes": list(spec.classes), "task": spec.key}, path)
    table.drop(columns="_cols").to_csv(out_dir / "cv_comparison.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(m, indent=2, default=float))
    return TabularResult(spec.key, table.drop(columns="_cols"), f"{best.features} | {best.reduction} | {best.classifier}",
                         m, path)

"""Feature attributions for the classical (tabular) models: tree/linear importance and optional SHAP."""
from __future__ import annotations

import numpy as np


def tree_feature_importance(pipeline, columns: list[str]) -> dict[str, float] | None:
    """Impurity or coefficient importance when features are not projected (no PCA/LDA)."""
    names = [n for n, _ in getattr(pipeline, "steps", [])]
    if "pca" in names or "lda" in names:
        return None  # importance would refer to components, not original features
    clf = pipeline.steps[-1][1]
    if hasattr(clf, "feature_importances_"):
        return dict(zip(columns, map(float, clf.feature_importances_)))
    if hasattr(clf, "coef_"):
        return dict(zip(columns, map(float, np.abs(clf.coef_).mean(0))))
    return None


def shap_values(pipeline, X_background: np.ndarray, X: np.ndarray, columns: list[str]):
    """Optional SHAP explanation (pip install shap). Returns per-feature contributions for X[0]."""
    import shap

    f = lambda Z: pipeline.predict_proba(Z)[:, 1]  # noqa: E731
    expl = shap.Explainer(f, X_background, feature_names=columns)
    sv = expl(X[:1])
    return dict(zip(columns, map(float, sv.values[0])))

"""PCA / LDA pipelines and the feature-representation comparison (ablation) study.

PCA vs LDA, in one paragraph (also shown in the app):
  PCA is *unsupervised*: it rotates the feature space onto directions of maximum total
  variance, ignoring labels. In radiomics/CNN features the largest variance is often lesion
  size, scanner/kernel or slice thickness — not necessarily what separates benign from
  malignant. LDA is *supervised*: it finds at most (K−1) directions that maximise
  between-class variance relative to within-class variance, so with 2 classes it yields a
  single discriminant axis. LDA needs a well-conditioned within-class covariance; with 512
  CNN features and a few hundred patients that matrix is singular, which is why we apply
  PCA first (variance-preserving compression) and LDA second (class separation), with
  shrinkage as a safeguard.

All transforms (scaler, PCA, LDA) live inside an sklearn Pipeline, so they are re-fitted
inside every cross-validation fold — fitting PCA on the full dataset before CV would leak
test-set information.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PCA_VS_LDA_TEXT = __doc__.split("PCA vs LDA, in one paragraph (also shown in the app):")[1].split("All transforms")[0].strip()


class LDATransform(BaseEstimator, TransformerMixin):
    """LDA used as a supervised projection (transform) inside a pipeline."""

    def __init__(self, n_components: int | None = None, shrinkage: str | float | None = "auto"):
        self.n_components, self.shrinkage = n_components, shrinkage

    def fit(self, X, y):
        solver = "eigen" if self.shrinkage is not None else "svd"
        self.lda_ = LinearDiscriminantAnalysis(n_components=self.n_components, solver=solver,
                                               shrinkage=self.shrinkage).fit(X, y)
        return self

    def transform(self, X):
        return self.lda_.transform(X)


def make_classifier(name: str, seed: int = 0, class_weight: str | None = "balanced"):
    if name == "logistic":
        return LogisticRegression(max_iter=5000, C=1.0, class_weight=class_weight)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight=class_weight,
                                      random_state=seed, n_jobs=-1)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                              class_weight=class_weight, random_state=seed)
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "xgboost":
        from xgboost import XGBClassifier  # optional

        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
    raise ValueError(name)


def build_pipeline(reduction: str, clf: str = "logistic", pca_var: float = 0.95, seed: int = 0) -> Pipeline:
    """reduction ∈ {none, pca, lda, pca_lda}."""
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if reduction in ("pca", "pca_lda"):
        steps.append(("pca", PCA(n_components=pca_var, svd_solver="full", random_state=seed)))
    if reduction in ("lda", "pca_lda"):
        steps.append(("lda", LDATransform()))
    steps.append(("clf", make_classifier(clf, seed)))
    return Pipeline(steps)


@dataclass
class FeatureSet:
    name: str
    columns: list[str]


def ablation_configs(radiomic_cols, deep_cols, clinical_cols) -> list[tuple[str, list[str], str]]:
    """(model name, feature columns, reduction) — mirrors the study design in the README."""
    cfgs = [
        ("A  Radiomics only", radiomic_cols, "none"),
        ("A' Radiomics + LDA", radiomic_cols, "lda"),
        ("C  Radiomics + PCA", radiomic_cols, "pca"),
    ]
    if deep_cols:
        cfgs += [
            ("B  CNN features only", deep_cols, "none"),
            ("B' CNN + PCA", deep_cols, "pca"),
            ("D  CNN + PCA + LDA", deep_cols, "pca_lda"),
            ("G  Radiomics + CNN + PCA", radiomic_cols + deep_cols, "pca"),
        ]
    if clinical_cols:
        if deep_cols:
            cfgs.append(("E  CNN + clinical", deep_cols + clinical_cols, "pca"))
            cfgs.append(("F  CNN + radiomics + clinical", deep_cols + radiomic_cols + clinical_cols, "pca"))
        cfgs.append(("H  Radiomics + clinical", radiomic_cols + clinical_cols, "none"))
    return cfgs


def fit_projections(X_train, y_train, X_eval, pca_var: float = 0.95):
    """Fit scaler+PCA and scaler+PCA+LDA on training data; project evaluation data for plotting."""
    base = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]).fit(X_train)
    Xt, Xe = base.transform(X_train), base.transform(X_eval)
    pca2 = PCA(n_components=2).fit(Xt)
    pca_full = PCA(n_components=pca_var, svd_solver="full").fit(Xt)
    lda = LDATransform().fit(pca_full.transform(Xt), y_train)
    return {
        "pca2_eval": pca2.transform(Xe),
        "pca_explained": pca2.explained_variance_ratio_.tolist(),
        "pca_n_components_95": int(pca_full.n_components_),
        "lda_eval": lda.transform(pca_full.transform(Xe)),
        "cumulative_variance": np.cumsum(PCA().fit(Xt).explained_variance_ratio_).tolist(),
    }


def clone_pipeline(p):
    return clone(p)

"""Classical-ML study on LIDC malignancy-suspicion: radiomics vs CNN features, with/without PCA/LDA.

python training/run_pca_lda.py

Uses the SAME patient-level split as the CNN (training/validation patients = development set
for grouped 5-fold CV; test patients evaluated once). Deep features for test patients come
from a CNN that never saw them. Writes models/classical_study/results.json (read by the app),
a CSV table and PCA/LDA/comparison figures.
"""
import _bootstrap  # noqa: F401

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ctcancer.classical.dimred import PCA_VS_LDA_TEXT, ablation_configs, build_pipeline, fit_projections
from ctcancer.config import cli_parser, load_config, set_seed
from ctcancer.data.splits import load_split
from ctcancer.evaluation.metrics import binary_metrics, bootstrap_ci
from ctcancer.evaluation.plots import feature_importance, model_comparison, projection_plot
from ctcancer.explain.gradcam import tree_feature_importance
from ctcancer.logging_utils import get_logger

log = get_logger("run_pca_lda")


def main():
    p = cli_parser(__doc__)
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    set_seed(cfg.seed)
    f = cfg.path("features_dir") / "lidc.csv"
    if not f.exists():
        raise SystemExit("[missing data] run training/extract_features.py --dataset lidc first.")
    df = pd.read_csv(f)
    df = df[df.label.notna()].copy()
    df["label"] = df.label.astype(int)
    split = load_split(cfg.path("splits_dir") / "lidc.json")
    dev = df[df.patient_id.astype(str).isin(split["train"] + split["val"])].reset_index(drop=True)
    test = df[df.patient_id.astype(str).isin(split["test"])].reset_index(drop=True)
    deep = [c for c in df.columns if c.startswith("deep_")]
    meta = {"lesion_id", "patient_id", "label", "dataset", "n_readers", "malignancy_mean", "malignancy_ratings",
            "diameter_mm_pylidc"}
    rad = [c for c in df.columns if c not in meta and not c.startswith("deep_") and pd.api.types.is_numeric_dtype(df[c])]
    if not deep:
        log.warning("No CNN features found — CNN arms of the study are skipped.")
    cv = StratifiedGroupKFold(cfg["classical"]["cv_folds"], shuffle=True, random_state=cfg.seed)
    rows = []
    for name, cols, red in ablation_configs(rad, deep, []):
        for clf in ["logistic", "random_forest"] + (["lda"] if red == "none" else []):
            if red in ("lda", "pca_lda") and clf != "logistic":
                continue
            cvs = []
            for tr, va in cv.split(dev, dev.label, groups=dev.patient_id):
                pipe = build_pipeline(red, clf, cfg["classical"]["pca_variance"], cfg.seed)
                pipe.fit(dev.loc[tr, cols].to_numpy(float), dev.label[tr])
                cvs.append(roc_auc_score(dev.label[va], pipe.predict_proba(dev.loc[va, cols].to_numpy(float))[:, 1]))
            pipe = build_pipeline(red, clf, cfg["classical"]["pca_variance"], cfg.seed)
            pipe.fit(dev[cols].to_numpy(float), dev.label)
            pt = pipe.predict_proba(test[cols].to_numpy(float))[:, 1]
            m = binary_metrics(test.label, pt)
            _, lo, hi = bootstrap_ci(roc_auc_score, test.label.to_numpy(), pt, test.patient_id.to_numpy(),
                                     cfg["evaluation"]["bootstrap_iterations"], seed=cfg.seed)
            n_comp = pipe.named_steps["pca"].n_components_ if "pca" in pipe.named_steps else None
            rows.append({"model": f"{name} · {clf}", "features": name, "reduction": red, "classifier": clf,
                         "n_input_features": len(cols), "n_pca_components": n_comp,
                         "cv_auc_mean": float(np.mean(cvs)), "cv_auc_std": float(np.std(cvs)),
                         "roc_auc": m["roc_auc"], "roc_auc_lo": lo, "roc_auc_hi": hi, "pr_auc": m["pr_auc"],
                         "sensitivity": m["recall_sensitivity"], "specificity": m["specificity"], "f1": m["f1"],
                         "brier": m["brier"]})
            if name.startswith("A ") and clf == "random_forest":
                imp = tree_feature_importance(pipe, cols)
                if imp:
                    feature_importance(list(imp), np.array(list(imp.values())),
                                       cfg.path("figures_dir") / "radiomics_rf_importance.png",
                                       "Radiomics — random-forest importance (dev set)")
    res = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
    out = cfg.path("models_dir") / "classical_study"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results.csv", index=False)
    figs = cfg.path("figures_dir")
    model_comparison(res.to_dict("records"), "roc_auc", figs / "ablation_comparison.png",
                     "LIDC suspicion: test ROC-AUC by representation (95% CI)")
    proj = {}
    for label, cols in (("radiomics", rad), ("cnn", deep)):
        if not cols:
            continue
        P = fit_projections(dev[cols].to_numpy(float), dev.label.to_numpy(), test[cols].to_numpy(float),
                            cfg["classical"]["pca_variance"])
        ev = P["pca_explained"]
        projection_plot(P["pca2_eval"], test.label.to_numpy(), ["likely benign", "suspicious"],
                        figs / f"pca_{label}.png", f"PCA of {label} features (test patients)",
                        (f"PC1 ({ev[0]:.0%} var)", f"PC2 ({ev[1]:.0%} var)"))
        projection_plot(P["lda_eval"], test.label.to_numpy(), ["likely benign", "suspicious"],
                        figs / f"lda_{label}.png", f"LDA of {label} features (after PCA, test patients)", ("LD1",))
        proj[label] = {"pca2": P["pca2_eval"].tolist(), "lda": P["lda_eval"].tolist(), "pca_explained": ev,
                       "pca_n_components_95": P["pca_n_components_95"],
                       "cumulative_variance": P["cumulative_variance"][:100]}
    (out / "results.json").write_text(json.dumps({
        "task": "LIDC malignancy suspicion (lesion level)", "explanation": PCA_VS_LDA_TEXT,
        "n_dev_patients": int(dev.patient_id.nunique()), "n_test_patients": int(test.patient_id.nunique()),
        "results": res.to_dict("records"), "projections": proj, "test_labels": test.label.tolist(),
    }, default=float))
    log.info("\n%s", res[["model", "cv_auc_mean", "roc_auc", "roc_auc_lo", "roc_auc_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()

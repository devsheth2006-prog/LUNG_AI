"""Evaluation metrics with bootstrap confidence intervals.

Bootstrap resampling is done over *patients* when patient IDs are supplied (lesions from the
same patient are resampled together), otherwise over samples.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage
from sklearn import metrics as skm


# ------------------------------------------------------------------ classification
def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)), "prevalence": float(y_true.mean()),
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "precision": skm.precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "f1": skm.f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
        "brier": skm.brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "threshold": threshold,
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = skm.roc_auc_score(y_true, y_prob)
        out["pr_auc"] = skm.average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def multiclass_metrics(y_true, prob: np.ndarray, class_names: list[str]) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = prob.argmax(1)
    out = {
        "n": int(len(y_true)), "accuracy": float(skm.accuracy_score(y_true, pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(skm.f1_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": skm.confusion_matrix(y_true, pred, labels=list(range(len(class_names)))).tolist(),
        "class_names": class_names,
        "per_class": skm.classification_report(y_true, pred, labels=list(range(len(class_names))),
                                               target_names=class_names, output_dict=True, zero_division=0),
    }
    try:
        out["macro_roc_auc_ovr"] = float(skm.roc_auc_score(y_true, prob, multi_class="ovr", average="macro",
                                                          labels=list(range(len(class_names)))))
    except ValueError:
        out["macro_roc_auc_ovr"] = float("nan")
    return out


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(y_prob[sel].mean() - y_true[sel].mean())
    return float(ece)


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def youden_threshold(y_true, y_prob) -> float:
    """Operating point chosen on the VALIDATION set, then frozen for the test set."""
    fpr, tpr, thr = skm.roc_curve(y_true, y_prob)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))


def bootstrap_ci(metric_fn: Callable, y_true, y_score, groups=None, n_boot: int = 1000, ci: float = 0.95,
                 seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile CI. Resamples patients (groups) when given."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    point = metric_fn(y_true, y_score)
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in uniq}
    stats = []
    for _ in range(n_boot):
        sel = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y_true[sel])) < 2 and y_true.ndim == 1:
            continue
        try:
            stats.append(metric_fn(y_true[sel], y_score[sel]))
        except ValueError:
            continue
    if not stats:
        return float(point), float("nan"), float("nan")
    a = (1 - ci) / 2
    return float(point), float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


# ------------------------------------------------------------------ segmentation
def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * (pred & gt).sum() / s)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    u = (pred | gt).sum()
    return 1.0 if u == 0 else float((pred & gt).sum() / u)


def hausdorff95(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """95th-percentile symmetric surface distance in mm."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    sp = spacing[-pred.ndim:]

    def surface(m):
        return m & ~ndimage.binary_erosion(m)

    sp_, sg = surface(pred), surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=sp)
    dt_p = ndimage.distance_transform_edt(~sp_, sampling=sp)
    d = np.concatenate([dt_g[sp_], dt_p[sg]])
    return float(np.percentile(d, 95))


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict[str, float]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum()
    tn = (~pred & ~gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    return {"dice": dice(pred, gt), "iou": iou(pred, gt),
            "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
            "hd95_mm": hausdorff95(pred, gt, spacing)}


# ------------------------------------------------------------------ regression
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"mae": float(skm.mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(skm.mean_squared_error(y_true, y_pred))),
            "r2": float(skm.r2_score(y_true, y_pred))}

"""Static figures (matplotlib) for the evaluation report / presentation slides."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import metrics as skm  # noqa: E402

PALETTE = ["#2a6f97", "#c8553d", "#588157", "#8a5a83", "#d4a017", "#5c677d"]
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def roc_pr_calibration(curves: dict[str, tuple], path, n_bins: int = 10):
    """curves: name -> (y_true, y_prob). Writes ROC, PR and reliability panels side by side."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for (name, (y, p)), c in zip(curves.items(), PALETTE):
        fpr, tpr, _ = skm.roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=c, label=f"{name} (AUC {skm.roc_auc_score(y, p):.3f})")
        pr, rc, _ = skm.precision_recall_curve(y, p)
        ax[1].plot(rc, pr, color=c, label=f"{name} (AP {skm.average_precision_score(y, p):.3f})")
        from sklearn.calibration import calibration_curve

        fp, mp = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax[2].plot(mp, fp, "o-", color=c, label=name)
    ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax[2].plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration")
    for a, (xl, yl, t) in zip(ax, [("1 − specificity", "sensitivity", "ROC"),
                                   ("recall", "precision", "Precision–recall"),
                                   ("mean predicted probability", "observed frequency", "Calibration")]):
        a.set(xlabel=xl, ylabel=yl, title=t, xlim=(0, 1), ylim=(0, 1.02))
        a.legend(fontsize=8, loc="lower right" if t != "Precision–recall" else "lower left")
    return _save(fig, path)


def confusion_matrix_plot(cm, class_names, path, title="Confusion matrix"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4 + 0.4 * len(class_names), 3.6 + 0.4 * len(class_names)))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)), xticklabels=class_names,
           yticklabels=class_names, xlabel="predicted", ylabel="reference label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, path)


def training_curves(history: list[dict], path, title="Training"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = [h["epoch"] for h in history]
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train", color=PALETTE[0])
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation", color=PALETTE[1])
    ax[0].set(title=f"{title} — loss", xlabel="epoch")
    key = next((k for k in ("val_auc", "val_dice", "val_metric") if k in history[0]), None)
    if key:
        ax[1].plot(ep, [h[key] for h in history], color=PALETTE[2])
        ax[1].set(title=f"{title} — {key}", xlabel="epoch")
    ax[0].legend()
    return _save(fig, path)


def projection_plot(Z: np.ndarray, y: np.ndarray, class_names, path, title: str, axis_labels=("1", "2")):
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    if Z.shape[1] == 1:  # LDA with 2 classes -> 1-D: show class densities
        for k, c in zip(np.unique(y), PALETTE):
            ax.hist(Z[y == k, 0], bins=30, alpha=0.55, color=c, label=class_names[int(k)], density=True)
        ax.set(xlabel=axis_labels[0], ylabel="density")
    else:
        for k, c in zip(np.unique(y), PALETTE):
            ax.scatter(Z[y == k, 0], Z[y == k, 1], s=10, alpha=0.6, color=c, label=class_names[int(k)])
        ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1])
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def model_comparison(results: list[dict], metric: str, path, title="Model comparison"):
    """results: [{'model':..., metric:..., f'{metric}_lo':..., f'{metric}_hi':...}]"""
    names = [r["model"] for r in results]
    vals = np.array([r[metric] for r in results])
    lo = np.array([r.get(f"{metric}_lo", np.nan) for r in results])
    hi = np.array([r.get(f"{metric}_hi", np.nan) for r in results])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.5))
    yy = np.arange(len(names))
    err = np.vstack([vals - lo, hi - vals]) if np.isfinite(lo).all() else None
    ax.barh(yy, vals, xerr=err, color=PALETTE[0], alpha=0.85, capsize=3)
    ax.set_yticks(yy, names)
    ax.invert_yaxis()
    ax.set(xlabel=metric, title=title, xlim=(0, 1))
    return _save(fig, path)


def feature_importance(names, values, path, title="Feature importance", top: int = 20):
    order = np.argsort(np.abs(values))[::-1][:top]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(order) + 1.2))
    ax.barh(range(len(order)), np.asarray(values)[order], color=PALETTE[2])
    ax.set_yticks(range(len(order)), [names[i] for i in order])
    ax.invert_yaxis()
    ax.set_title(title)
    return _save(fig, path)

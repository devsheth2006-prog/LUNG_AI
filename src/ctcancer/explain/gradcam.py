"""Grad-CAM and Grad-CAM++ for 2-D and 3-D CNNs, plus tree/linear feature attributions.

Interpretation (shown verbatim in the UI): heatmaps highlight *regions contributing most
strongly to this model prediction*. They are coarse (computed on the last convolutional
feature map, then upsampled), can highlight context outside the lesion, and do not prove
where a tumour is.
"""
from __future__ import annotations

import numpy as np

INTERPRETATION = ("Regions contributing most strongly to this model prediction (Grad-CAM). "
                  "Heatmaps are coarse, reflect what the model used rather than ground truth, "
                  "and do not prove the presence or location of a tumour.")


class GradCAM:
    def __init__(self, model, target_layer, plus_plus: bool = False):
        self.model, self.layer, self.pp = model, target_layer, plus_plus
        self._acts = self._grads = None
        # forward hook + tensor gradient hook (avoids full-backward-hook conflicts with in-place ops)
        self._h = [target_layer.register_forward_hook(self._fwd)]

    def _fwd(self, _m, _i, out):
        self._acts = out.detach()
        if out.requires_grad:
            out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad.detach()

    def remove(self):
        for h in self._h:
            h.remove()

    def __call__(self, x, class_idx: int | None = None) -> tuple[np.ndarray, int, float]:
        import torch
        import torch.nn.functional as F

        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1)[0])
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
        A, G = self._acts[0], self._grads[0]                 # (C, *spatial)
        spatial = tuple(range(1, A.ndim))
        if self.pp:
            g2, g3 = G ** 2, G ** 3
            denom = 2 * g2 + (A * g3).sum(dim=spatial, keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            w = (alpha * F.relu(G)).sum(dim=spatial)
        else:
            w = G.mean(dim=spatial)
        cam = F.relu((w.view(-1, *([1] * (A.ndim - 1))) * A).sum(0))
        cam = cam[None, None]
        mode = "bilinear" if cam.ndim == 4 else "trilinear"
        cam = F.interpolate(cam, size=x.shape[2:], mode=mode, align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        prob = float(logits.softmax(1)[0, class_idx])
        return cam.cpu().numpy(), class_idx, prob


def overlay_heatmap(gray01: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB uint8 overlay of a [0,1] heatmap on a [0,1] grayscale image (for display only)."""
    import matplotlib

    cmap = matplotlib.colormaps["inferno"]
    heat = cmap(np.clip(cam, 0, 1))[..., :3]
    base = np.repeat(np.clip(gray01, 0, 1)[..., None], 3, -1)
    out = (1 - alpha * cam[..., None]) * base + alpha * cam[..., None] * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


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

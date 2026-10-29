"""Shared PyTorch training loop: early stopping on a validation metric, staged unfreezing,
best-checkpoint saving and per-epoch tracking."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .logging_utils import get_logger

log = get_logger(__name__)


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def fit(model, train_loader, val_loader, loss_fn, device, *, epochs: int, lr: float, weight_decay: float,
        val_metric: Callable[[np.ndarray, np.ndarray], float], patience: int = 6, tracker=None,
        freeze_epochs: int = 0, out_path: str | Path | None = None, metric_name: str = "val_auc") -> tuple[torch.nn.Module, list[dict]]:
    """Train with AdamW + cosine schedule; returns the best model (by val_metric) and history."""
    model.to(device)
    loss_fn.to(device)
    staged = hasattr(model, "set_stage") and freeze_epochs > 0
    if staged:
        model.set_stage(1)

    def make_opt():
        params = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("net.")],
                   "lr": lr},
                  {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("net.")],
                   "lr": lr * 0.1}]
        params = [g for g in params if g["params"]]
        return torch.optim.AdamW(params, weight_decay=weight_decay)

    opt = make_opt()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best, best_state, wait, history = -np.inf, None, 0, []
    for ep in range(1, epochs + 1):
        if staged and ep == freeze_epochs + 1:
            model.set_stage(2)
            opt = make_opt()
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - freeze_epochs, 1))
            log.info("Unfroze last backbone stage for fine-tuning")
        model.train()
        t0, tl, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(x)
            n += len(x)
        sched.step()
        logits, yv = predict(model, val_loader, device)
        vl = float(loss_fn(torch.from_numpy(logits).to(device), torch.from_numpy(yv).to(device)).item())
        vm = float(val_metric(yv, logits))
        rec = {"epoch": ep, "train_loss": tl / max(n, 1), "val_loss": vl, metric_name: vm, "sec": time.time() - t0}
        history.append(rec)
        if tracker:
            tracker.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=ep)
        log.info("epoch %d  train_loss %.4f  val_loss %.4f  %s %.4f", ep, rec["train_loss"], vl, metric_name, vm)
        if vm > best + 1e-4:
            best, best_state, wait = vm, copy.deepcopy(model.state_dict()), 0
            if out_path:
                torch.save({"state_dict": best_state, "epoch": ep, metric_name: vm}, out_path)
        else:
            wait += 1
            if wait >= patience:
                log.info("Early stopping at epoch %d (best %s %.4f)", ep, metric_name, best)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

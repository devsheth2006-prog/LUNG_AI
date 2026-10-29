"""Train a lung or tumour/nodule segmentation U-Net and evaluate it on held-out patients.

python training/train_segmentation.py --target lung     # LUNA16 lung masks (2-D U-Net, lite)
python training/train_segmentation.py --target tumor    # LIDC consensus + NSCLC GTV masks
Metrics on the test patients: Dice, IoU, sensitivity, specificity, HD95 (mean + patient-bootstrap 95% CI).
"""
import _bootstrap  # noqa: F401

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ctcancer.config import cli_parser, load_config, resolve_device, set_seed
from ctcancer.data.augment import SliceAugment
from ctcancer.data.datasets import SegSliceDataset
from ctcancer.data.splits import load_split
from ctcancer.evaluation.metrics import segmentation_metrics
from ctcancer.evaluation.plots import training_curves
from ctcancer.inference.registry import write_card
from ctcancer.logging_utils import get_logger
from ctcancer.models.unet import DiceCELoss, build_segmentation_model
from ctcancer.tracking import Tracker, file_hash
from ctcancer.training_utils import fit

log = get_logger("train_segmentation")


def load_tables(cfg, target):
    proc, splits = cfg.path("processed_root"), cfg.path("splits_dir")
    if target == "lung":
        sources = [("luna16_lung", "lung_slices.csv")]
    else:
        sources = [(d, "lesions.csv") for d in ("lidc", "nsclc_radiomics", "nsclc_radiogenomics")
                   if (proc / d / "lesions.csv").exists()]
    if not sources or not all((proc / d / f).exists() for d, f in sources):
        raise SystemExit("[missing data] run training/prepare_data.py first (see README › Data preparation).")
    parts = {"train": [], "val": [], "test": []}
    for d, f in sources:
        t = pd.read_csv(proc / d / f)
        prefix = f"{d}/"
        if "path" in t:
            t["path"] = prefix + t["path"]
        if "patch_path" in t:
            t["patch_path"] = prefix + t["patch_path"]
        sp = load_split(splits / f"{d}.json")
        for k in parts:
            parts[k].append(t[t.patient_id.astype(str).isin(sp[k])])
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}, sources


def main():
    p = cli_parser(__doc__)
    p.add_argument("--target", choices=["lung", "tumor"], required=True)
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    tables, sources = load_tables(cfg, a.target)
    scfg = cfg["segmentation"]["lung" if a.target == "lung" else "tumor"]
    arch = scfg["model_name"] if scfg["model_name"] == "unet2d" else "unet2d"  # cached data is 2-D slices
    kind = "lung" if a.target == "lung" else "tumor"
    root = cfg.path("processed_root")
    tr = SegSliceDataset(tables["train"], root, kind, SliceAugment(cfg.seed))
    va = SegSliceDataset(tables["val"], root, kind)
    te = SegSliceDataset(tables["test"], root, kind)
    bs, nw = cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    dl = lambda ds, sh: DataLoader(ds, bs, shuffle=sh, num_workers=nw)  # noqa: E731
    model = build_segmentation_model(arch, 1, 2, scfg["base_channels"])
    task = "lung_segmentation" if a.target == "lung" else "tumor_segmentation"
    out_dir = cfg.path("models_dir") / task
    out_dir.mkdir(parents=True, exist_ok=True)

    def val_dice(y, logits):
        pred = logits.argmax(1)
        return float(np.mean([segmentation_metrics(p, t)["dice"] for p, t in zip(pred, y) if t.any() or p.any()] or [0]))

    tracker = Tracker(cfg["tracking"]["experiment_name"], f"{task}-{arch}", cfg.path("mlruns"), cfg["tracking"]["backend"])
    with tracker.run():
        tracker.log_params({"task": task, "arch": arch, **cfg["training"], "base_channels": scfg["base_channels"]})
        tracker.set_tags({"dataset_version": ",".join(file_hash(cfg.path("splits_dir") / f"{d}.json") for d, _ in sources)})
        model, hist = fit(model, dl(tr, True), dl(va, False), DiceCELoss(), device,
                          epochs=cfg["training"]["epochs"], lr=cfg["training"]["lr"],
                          weight_decay=cfg["training"]["weight_decay"], val_metric=val_dice,
                          patience=cfg["training"]["early_stopping_patience"], tracker=tracker,
                          out_path=out_dir / "model.pt", metric_name="val_dice")
        # ---- single, final evaluation on the held-out TEST patients
        model.eval()
        per = []
        with torch.no_grad():
            for i in range(len(te)):
                x, y = te[i]
                pred = model(x[None].to(device)).argmax(1)[0].cpu().numpy()
                if y.numpy().any() or pred.any():
                    per.append(segmentation_metrics(pred, y.numpy(), (1.0, 1.0)))
        dfm = pd.DataFrame(per)
        rng = np.random.default_rng(cfg.seed)
        test = {}
        for k in ["dice", "iou", "sensitivity", "specificity", "hd95_mm"]:
            v = dfm[k].dropna().to_numpy()
            boots = [rng.choice(v, len(v)).mean() for _ in range(cfg["evaluation"]["bootstrap_iterations"])] if len(v) else [np.nan]
            test[k] = float(v.mean()) if len(v) else float("nan")
            test[f"{k}_ci"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        test["n_test_slices"] = len(dfm)
        test["n_test_patients"] = int(tables["test"].patient_id.nunique())
        tracker.log_metrics({f"test_{k}": v for k, v in test.items() if isinstance(v, float)})
        fig = training_curves(hist, cfg.path("figures_dir") / f"{task}_curves.png", task)
        tracker.log_artifact(fig)
    write_card(task, {"architecture": arch, "base_channels": scfg["base_channels"],
                      "input_size": cfg["preprocessing"]["slice_size"], "patch_vox": 96, "version": "1",
                      "datasets": [d for d, _ in sources],
                      "n_patients": {k: int(v.patient_id.nunique()) for k, v in tables.items()},
                      "test_metrics": test, "history": hist, "metric_note":
                      "2-D slice-level metrics on held-out patients; HD95 in pixels≈mm at the cached resolution."},
               cfg.path("models_dir"))
    log.info("Test metrics: %s", json.dumps({k: v for k, v in test.items() if not k.endswith('_ci')}, default=float))


if __name__ == "__main__":
    main()

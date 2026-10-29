"""Train and compare lesion classifiers (transfer learning), calibrate, evaluate once on test.

python training/train_classifier.py --task malignancy          # LIDC-IDRI radiologist-suspicion label
python training/train_classifier.py --task nodule_detection    # LUNA16 candidate FP reduction
python training/train_classifier.py --task malignancy --set classifier.architectures=[resnet18]

Protocol: architectures in `classifier.architectures` are trained on the TRAIN patients,
early-stopped and temperature-calibrated on VAL patients, and the operating threshold is chosen
on VAL (Youden J). The model to deploy is selected by VAL ROC-AUC. Every architecture is then
evaluated once on the TEST patients and all results are reported (not only the winner).
"""
import _bootstrap  # noqa: F401

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from ctcancer.config import cli_parser, load_config, resolve_device, set_seed
from ctcancer.data.augment import PatchAugment
from ctcancer.data.datasets import LesionPatchDataset
from ctcancer.data.splits import load_split
from ctcancer.evaluation.metrics import binary_metrics, bootstrap_ci, youden_threshold
from ctcancer.evaluation.plots import confusion_matrix_plot, model_comparison, roc_pr_calibration, training_curves
from ctcancer.inference.registry import write_card
from ctcancer.logging_utils import get_logger
from ctcancer.models.classifiers import FocalLoss, TemperatureScaler, build_classifier
from ctcancer.risk.tasks import TASKS
from ctcancer.tracking import Tracker, file_hash
from ctcancer.training_utils import class_weights, fit, predict

log = get_logger("train_classifier")
SOURCES = {"malignancy": ("lidc", "lesions.csv", 64), "nodule_detection": ("luna16_candidates", "candidates.csv", 48)}


def main():
    p = cli_parser(__doc__)
    p.add_argument("--task", choices=list(SOURCES), required=True)
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    d, csv, patch = SOURCES[a.task]
    root = cfg.path("processed_root") / d
    if not (root / csv).exists():
        raise SystemExit(f"[missing data] {root / csv} not found — run training/prepare_data.py --dataset {d}.")
    t = pd.read_csv(root / csv)
    t = t[t.label.notna()].copy()
    t["label"] = t.label.astype(int)
    # patch size the model is actually trained on
    patch = int(np.load(root / t.patch_path.iloc[0])["hu"].shape[-1]) if len(t) else patch
    split_file = cfg.path("splits_dir") / f"{d}.json"
    split = load_split(split_file)
    parts = {k: t[t.patient_id.astype(str).isin(v)].reset_index(drop=True) for k, v in split.items()}
    log.info("patients/lesions: %s", {k: (v.patient_id.nunique(), len(v)) for k, v in parts.items()})

    ccfg, tcfg = cfg["classifier"], cfg["training"]
    out_dir = cfg.path("models_dir") / a.task
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = cfg.path("figures_dir")
    results, curves, best = [], {}, None
    for arch in ccfg["architectures"]:
        set_seed(cfg.seed)
        mode = "3d" if arch == "cnn3d" else "2.5d"
        aug = PatchAugment(seed=cfg.seed)
        ds = {k: LesionPatchDataset(v, root, "label", mode, aug if k == "train" else None) for k, v in parts.items()}
        dl = {k: DataLoader(v, tcfg["batch_size"], shuffle=(k == "train"), num_workers=tcfg["num_workers"])
              for k, v in ds.items()}
        w = class_weights(parts["train"].label.to_numpy(), 2)
        loss = FocalLoss(tcfg["focal_gamma"], w) if tcfg["loss"] == "focal" else torch.nn.CrossEntropyLoss(
            weight=w if tcfg["loss"] == "weighted_ce" else None)
        model = build_classifier(arch, 2, ccfg["pretrained"])
        auc = lambda y, lg: roc_auc_score(y, torch.from_numpy(lg).softmax(1)[:, 1].numpy())  # noqa: E731
        tracker = Tracker(cfg["tracking"]["experiment_name"], f"{a.task}-{arch}", cfg.path("mlruns"),
                          cfg["tracking"]["backend"])
        with tracker.run():
            tracker.log_params({"task": a.task, "arch": arch, "input_mode": mode, **tcfg})
            tracker.set_tags({"dataset_version": file_hash(split_file), "model_version": "1"})
            model, hist = fit(model, dl["train"], dl["val"], loss, device, epochs=tcfg["epochs"], lr=tcfg["lr"],
                              weight_decay=tcfg["weight_decay"], val_metric=auc,
                              patience=tcfg["early_stopping_patience"], tracker=tracker,
                              freeze_epochs=tcfg["freeze_backbone_epochs"] if ccfg["pretrained"] else 0,
                              out_path=out_dir / f"{arch}.pt")
            # calibration + threshold on VALIDATION patients only
            lv, yv = predict(model, dl["val"], device)
            ts = TemperatureScaler().fit(torch.from_numpy(lv), torch.from_numpy(yv).long()) \
                if ccfg["calibrate"] == "temperature" else TemperatureScaler()
            pv = torch.from_numpy(lv / ts.temperature).softmax(1)[:, 1].numpy()
            thr = youden_threshold(yv, pv)
            val_auc = roc_auc_score(yv, pv)
            # final TEST evaluation
            lt, yt = predict(model, dl["test"], device)
            pt = torch.from_numpy(lt / ts.temperature).softmax(1)[:, 1].numpy()
            m = binary_metrics(yt, pt, thr)
            groups = parts["test"].patient_id.to_numpy()
            n_boot = cfg["evaluation"]["bootstrap_iterations"]
            for name, fn in (("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)):
                _, lo, hi = bootstrap_ci(fn, yt, pt, groups, n_boot, seed=cfg.seed)
                m[f"{name}_lo"], m[f"{name}_hi"] = lo, hi
            m.update({"val_roc_auc": float(val_auc), "temperature": ts.temperature})
            tracker.log_metrics({f"test_{k}": v for k, v in m.items() if isinstance(v, float)})
        results.append({"model": arch, **m, "history": hist})
        curves[arch] = (yt, pt)
        training_curves(hist, fig_dir / f"{a.task}_{arch}_curves.png", f"{a.task} {arch}")
        confusion_matrix_plot([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], list(TASKS[a.task].classes),
                              fig_dir / f"{a.task}_{arch}_confusion.png", f"{arch} (test, thr={thr:.2f})")
        if best is None or val_auc > best[1]:
            best = (arch, val_auc, model, ts.temperature, thr, m)

    roc_pr_calibration(curves, fig_dir / f"{a.task}_roc_pr_calibration.png")
    model_comparison(results, "roc_auc", fig_dir / f"{a.task}_model_comparison.png", f"{a.task}: test ROC-AUC (95% CI)")
    arch, _, model, temp, thr, m = best
    torch.save({"state_dict": model.state_dict()}, out_dir / "model.pt")
    write_card(a.task, {
        "architecture": arch, "version": "1", "classes": list(TASKS[a.task].classes), "input_mode":
        "3d" if arch == "cnn3d" else "2.5d", "patch_vox": patch, "temperature": temp, "threshold": thr,
        "datasets": [d], "selection": "best validation ROC-AUC",
        "n_patients": {k: int(v.patient_id.nunique()) for k, v in parts.items()},
        "n_lesions": {k: len(v) for k, v in parts.items()},
        "test_metrics": {k: v for k, v in m.items() if k != "history"},
        "comparison": [{k: v for k, v in r.items() if k != "history"} for r in results],
        "history": next(r["history"] for r in results if r["model"] == arch),
    }, cfg.path("models_dir"))
    pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in results]).to_csv(
        out_dir / "architecture_comparison.csv", index=False)
    log.info("Deployed %s. Test: %s", arch, json.dumps({k: round(v, 4) for k, v in m.items()
                                                        if isinstance(v, float)}))


if __name__ == "__main__":
    main()

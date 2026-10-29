"""Single-modality vs multimodal (CNN embedding + clinical) multi-task network.

python training/train_multimodal.py

Image branch: frozen malignancy-CNN embedding of the primary tumour.
Clinical branch: only variables present in the cohorts (age, sex, smoking, pack-years) with
missing-indicators and TRAIN-only imputation. Heads: one per task with enough labels
(subtype, n_involvement); losses are masked where a patient lacks that label.
Compares image-only vs image+clinical on the same patient hold-out split (study result; the
deployed patient-level heads are the calibrated tabular models from train_stage/metastasis).
"""
import _bootstrap  # noqa: F401

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from _patient_tasks import load_patient_features
from ctcancer.config import cli_parser, load_config, set_seed
from ctcancer.data.clinical import CLINICAL_FEATURES, clinical_matrix, fit_clinical_imputer
from ctcancer.data.splits import patient_holdout
from ctcancer.logging_utils import get_logger
from ctcancer.models.classifiers import MultimodalNet
from ctcancer.risk.tasks import TASKS

log = get_logger("train_multimodal")
HEADS = {"n_involvement": ("n_positive", 2), "subtype": ("histology", 3)}


def encode(df, task):
    col, _ = HEADS[task]
    if task == "subtype":
        m = {c: i for i, c in enumerate(TASKS["subtype"].classes)}
        return df[col].map(m).fillna(-1).astype(int).to_numpy()
    return df[col].fillna(-1).astype(int).to_numpy()


def train_eval(Xi_tr, Xc_tr, Y_tr, Xi_te, Xc_te, Y_te, heads, epochs, seed):
    torch.manual_seed(seed)
    net = MultimodalNet(Xi_tr.shape[1], Xc_tr.shape[1] if Xc_tr is not None else 0, heads)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    ti = torch.tensor(Xi_tr, dtype=torch.float32)
    tc = torch.tensor(Xc_tr, dtype=torch.float32) if Xc_tr is not None else None
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        out = net(ti, tc)
        loss = 0
        for k in heads:
            y = torch.tensor(Y_tr[k])
            m = y >= 0
            if m.any():
                loss = loss + F.cross_entropy(out[k][m], y[m])
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor(Xi_te, dtype=torch.float32),
                  torch.tensor(Xc_te, dtype=torch.float32) if Xc_te is not None else None)
    res = {}
    for k, n in heads.items():
        y, m = Y_te[k], Y_te[k] >= 0
        p = out[k].softmax(1).numpy()[m]
        try:
            res[k] = float(roc_auc_score(y[m], p[:, 1]) if n == 2 else
                           roc_auc_score(y[m], p, multi_class="ovr", labels=list(range(n))))
        except ValueError:
            res[k] = float("nan")
    return res


def main():
    p = cli_parser(__doc__)
    p.add_argument("--epochs", type=int, default=300)
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    set_seed(cfg.seed)
    df = load_patient_features(cfg)
    deep = [c for c in df.columns if c.startswith("deep_")]
    if not deep:
        raise SystemExit("[missing features] the malignancy CNN must be trained and extract_features re-run.")
    min_pos = cfg["risk"]["min_positive_labels"]
    heads = {}
    for k, (col, n) in HEADS.items():
        y = encode(df, k)
        counts = np.bincount(y[y >= 0], minlength=n)
        if (counts >= min_pos).all():
            heads[k] = n
        else:
            log.warning("head %s skipped: class counts %s", k, counts.tolist())
    if not heads:
        raise SystemExit("No task has enough labels for a multimodal head.")
    split = patient_holdout(df.assign(label=encode(df, next(iter(heads))).clip(0)), 0.2, seed=cfg.seed)
    tr, te = df[df.patient_id.isin(split["dev"])], df[df.patient_id.isin(split["test"])]
    sc = StandardScaler().fit(tr[deep])
    Xi_tr, Xi_te = sc.transform(tr[deep]), sc.transform(te[deep])
    fills = fit_clinical_imputer(tr, CLINICAL_FEATURES)
    Xc_tr, names = clinical_matrix(tr, fills)
    Xc_te, _ = clinical_matrix(te, fills)
    if Xc_tr.shape[1]:
        cs = StandardScaler().fit(Xc_tr)
        Xc_tr, Xc_te = cs.transform(Xc_tr), cs.transform(Xc_te)
    Y_tr = {k: encode(tr, k) for k in heads}
    Y_te = {k: encode(te, k) for k in heads}
    results = {"image_only": train_eval(Xi_tr, None, Y_tr, Xi_te, None, Y_te, heads, a.epochs, cfg.seed)}
    if Xc_tr.shape[1]:
        results["image_plus_clinical"] = train_eval(Xi_tr, Xc_tr, Y_tr, Xi_te, Xc_te, Y_te, heads, a.epochs, cfg.seed)
    else:
        log.warning("No clinical variables available in the loaded cohorts — multimodal arm skipped.")
    out = cfg.path("models_dir") / "multimodal_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"heads": heads, "clinical_variables": names, "test_auc": results,
                                                  "n_train": len(tr), "n_test": len(te)}, indent=2))
    log.info("Multimodal study (test AUC): %s", results)


if __name__ == "__main__":
    main()

"""Extract radiomic + deep CNN features for every cached lesion/tumour.

python training/extract_features.py --dataset lidc
python training/extract_features.py --dataset nsclc_radiomics
python training/extract_features.py --dataset nsclc_radiogenomics

Deep features = the penultimate-layer embedding of the *trained malignancy CNN* (ResNet18:
512-D, DenseNet121: 1024-D). If that model is not trained yet, only radiomic features are
written and downstream studies skip the CNN-feature arms (logged, never faked).
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from ctcancer.config import cli_parser, load_config
from ctcancer.data.datasets import patch_to_input
from ctcancer.features.radiomics import extract_radiomics
from ctcancer.inference.pipeline import ModelBank
from ctcancer.logging_utils import get_logger

log = get_logger("extract_features")


def main():
    p = cli_parser(__doc__)
    p.add_argument("--dataset", required=True, choices=["lidc", "nsclc_radiomics", "nsclc_radiogenomics"])
    p.add_argument("--pyradiomics", action="store_true", help="use PyRadiomics (IBSI) instead of built-in features")
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    root = cfg.path("processed_root") / a.dataset
    if not (root / "lesions.csv").exists():
        raise SystemExit(f"[missing data] run training/prepare_data.py --dataset {a.dataset} first.")
    table = pd.read_csv(root / "lesions.csv")
    bank = ModelBank(cfg)
    mal = bank.torch_model("malignancy")
    if mal is None:
        log.warning("Malignancy CNN not trained: deep features skipped (radiomics only).")
    else:
        import torch
    rows = []
    for i, r in table.iterrows():
        d = np.load(root / r.patch_path)
        hu, mask, sp = d["hu"].astype(np.float32), d["mask"], tuple(d["spacing"])
        if not mask.any():
            continue
        if a.pyradiomics:
            from ctcancer.features.radiomics import extract_pyradiomics

            feats = extract_pyradiomics(hu, mask, sp)
        else:
            feats = extract_radiomics(hu, mask, sp)
        row = {"lesion_id": r.lesion_id, "patient_id": r.patient_id, **feats}
        if mal is not None:
            model, card = mal
            pass  # no extra crop
            with torch.no_grad():
                x = torch.from_numpy(patch_to_input(hu, card.get("input_mode", "2.5d"))[None]).float()
                emb = model.embed(x.to(bank._dev()))[0].cpu().numpy()
            row.update({f"deep_{j:03d}": float(v) for j, v in enumerate(emb)})
        rows.append(row)
        if i % 100 == 0:
            log.info("%d/%d", i + 1, len(table))
    feats = pd.DataFrame(rows)
    label_cols = [c for c in table.columns if c not in ("patch_path",) and c not in feats.columns]
    out = feats.merge(table[["lesion_id"] + label_cols], on="lesion_id", how="left")
    dst = cfg.path("features_dir") / f"{a.dataset}.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    log.info("Wrote %s (%d rows, %d radiomic, %d deep features)", dst, len(out),
             sum(not c.startswith("deep_") for c in feats.columns) - 2, sum(c.startswith("deep_") for c in feats.columns))


if __name__ == "__main__":
    main()

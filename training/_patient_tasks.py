"""Shared driver for patient-level tabular tasks (used by train_stage_model / train_metastasis_model)."""
import _bootstrap  # noqa: F401

import pandas as pd

from ctcancer.data.clinical import CLINICAL_FEATURES
from ctcancer.inference.registry import write_card
from ctcancer.logging_utils import get_logger
from ctcancer.risk.tabular import InsufficientLabelsError, train_tabular_task
from ctcancer.risk.tasks import TASKS

log = get_logger("patient_tasks")


def load_patient_features(cfg) -> pd.DataFrame:
    parts = []
    for d in ("nsclc_radiomics", "nsclc_radiogenomics"):
        f = cfg.path("features_dir") / f"{d}.csv"
        if f.exists():
            t = pd.read_csv(f)
            t["patient_id"] = d + ":" + t.patient_id.astype(str)   # IDs unique across cohorts
            parts.append(t)
    if not parts:
        raise SystemExit("[missing data] run prepare_data.py + extract_features.py for nsclc_radiomics and/or "
                         "nsclc_radiogenomics first.")
    df = pd.concat(parts, ignore_index=True)
    # one primary tumour per patient
    return df.sort_values("volume_mm3", ascending=False).drop_duplicates("patient_id")


def run_tasks(cfg, tasks: list[str]) -> None:
    df = load_patient_features(cfg)
    deep = [c for c in df.columns if c.startswith("deep_")]
    clinical = [c for c in CLINICAL_FEATURES if c in df and df[c].notna().mean() > 0.5]
    skip = {"lesion_id", "patient_id", "dataset", "source", "staging_type", "histology", "t_category",
            "n_category", "n_positive", "m_positive", "distant_recurrence", "gtv_volume_cm3", "gtv_exceeds_patch"}
    rad = [c for c in df.columns if c not in skip and c not in deep and c not in clinical
           and c not in CLINICAL_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
    log.info("features: %d radiomic, %d deep, clinical=%s; patients=%d", len(rad), len(deep), clinical, len(df))
    for task in tasks:
        spec = TASKS[task]
        if spec.label_col not in df:
            log.warning("%s: label column '%s' not present in any loaded cohort", task, spec.label_col)
            write_card(task, {"refused": True, "reason": f"No '{spec.label_col}' labels in the prepared cohorts."}, cfg.path("models_dir"))
            continue
        try:
            r = train_tabular_task(df, spec, rad, deep, clinical, cfg.path("models_dir") / task, cfg, cfg.seed)
        except InsufficientLabelsError as exc:
            log.warning(str(exc))
            write_card(task, {"refused": True, "reason": str(exc)}, cfg.path("models_dir"))
            continue
        write_card(task, {"architecture": r.best_name, "version": "1", "classes": list(r.test_metrics.get(
            "class_names", spec.classes)) if "class_names" in r.test_metrics else list(spec.classes),
            "datasets": sorted(df.dataset.dropna().unique().tolist()) if "dataset" in df else [],
            "test_metrics": r.test_metrics, "comparison": r.cv_table.head(25).to_dict("records"),
            "staging_note": "Lung1 = clinical TNM; Radiogenomics = pathological TNM (mixed sources)."},
            cfg.path("models_dir"))
        log.info("%s -> %s", task, r.best_name)

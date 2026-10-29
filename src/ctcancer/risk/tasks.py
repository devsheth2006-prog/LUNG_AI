"""Single source of truth for every prediction task: what label it learns, from which dataset,
how the output must be worded, and its scientific caveats. The UI, report and training scripts
all read from here, so wording cannot drift into unsupported claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    level: str                       # lesion | patient | voxel
    label_col: str | None
    classes: tuple[str, ...]
    datasets: tuple[str, ...]
    label_kind: str                  # what the reference label actually is
    output_wording: str              # how the UI must phrase the output
    caveat: str
    supported: bool = True           # False -> no dataset can supervise it
    min_positive: int = 15
    extras: dict = field(default_factory=dict)


TASKS: dict[str, TaskSpec] = {t.key: t for t in [
    TaskSpec("lung_segmentation", "Lung segmentation", "voxel", None, ("background", "lung"),
             ("LUNA16",), "Automatically generated lung masks (LUNA16 seg-lungs)",
             "Lung region", "Reference masks are themselves algorithmic; Dice measures agreement with them."),
    TaskSpec("nodule_detection", "Nodule candidate classification", "lesion", "label", ("non-nodule", "nodule"),
             ("LUNA16",), "Nodules ≥3 mm accepted by ≥3 of 4 radiologists",
             "Model-estimated probability that a candidate is a nodule",
             "Detects nodules, not cancers. Trained on LIDC-derived scans only (single source, mostly 2000s scanners)."),
    TaskSpec("tumor_segmentation", "Tumour / nodule segmentation", "voxel", None, ("background", "lesion"),
             ("LIDC-IDRI", "NSCLC-Radiomics", "NSCLC-Radiogenomics"),
             "LIDC 50%-consensus nodule contours; NSCLC GTV contours",
             "Model-estimated lesion boundary", "GTV contours include radiotherapy margins conventions; inter-reader variability is large."),
    TaskSpec("malignancy", "Malignancy suspicion", "lesion", "label", ("likely benign", "suspicious"),
             ("LIDC-IDRI",), "Mean LIDC radiologist malignancy rating (<3 vs >3); NOT pathology",
             "Model-estimated malignancy probability (calibrated to radiologist suspicion ratings)",
             "Reference standard is radiologist opinion, not biopsy. Indeterminate nodules (mean rating 3) were excluded, "
             "which makes the task easier than real practice."),
    TaskSpec("subtype", "Histological subtype", "patient", "histology",
             ("adenocarcinoma", "squamous_cell_carcinoma", "other_nsclc"),
             ("NSCLC-Radiomics", "NSCLC-Radiogenomics"), "Pathology-reported histology",
             "Model-estimated subtype probabilities (NSCLC only)",
             "Only applicable to a known NSCLC primary tumour. Small-cell lung cancer is absent from training data and "
             "cannot be predicted. Imaging-based subtype prediction is known to be only moderately accurate."),
    TaskSpec("t_category", "T-category estimate", "patient", "t_category", ("T1", "T2", "T3", "T4"),
             ("NSCLC-Radiomics", "NSCLC-Radiogenomics"),
             "Clinical (Lung1) or pathological (Radiogenomics) T category — different TNM editions/sources",
             "Model-estimated T-category probability distribution — requires clinical confirmation",
             "T depends on invasion of structures (chest wall, mediastinum) that a lesion patch may not show. "
             "Labels mix clinical and pathological staging; TNM edition differs between cohorts."),
    TaskSpec("n_involvement", "Regional lymph-node involvement", "patient", "n_positive", ("N0", "N1–N3"),
             ("NSCLC-Radiomics", "NSCLC-Radiogenomics"), "N category from the clinical CSV (N0 vs N≥1)",
             "Model-estimated probability of regional nodal involvement — requires clinical confirmation",
             "Predicted from the primary tumour's appearance only; lymph nodes themselves are not segmented. "
             "PET/EBUS remain the clinical standard."),
    TaskSpec("m_distant", "Distant metastasis (M1)", "patient", "m_positive", ("M0", "M1"),
             ("NSCLC-Radiomics", "NSCLC-Radiogenomics"), "M category from the clinical CSV",
             "Model-estimated probability of distant metastasis — requires clinical confirmation",
             "M1 cases are rare in both cohorts. Training is refused below `risk.min_positive_labels` positives; "
             "a chest CT patch cannot show distant organs."),
    TaskSpec("distant_recurrence", "Distant recurrence after treatment (extension)", "patient", "distant_recurrence",
             ("no distant recurrence", "distant recurrence"), ("NSCLC-Radiogenomics",),
             "Recurrence location = distant (post-treatment follow-up)",
             "Model-estimated probability of later distant recurrence (research extension)",
             "This is a prognostic follow-up label, NOT baseline metastasis; censoring/follow-up length is ignored."),
    TaskSpec("organ_specific_metastasis", "Organ-specific metastasis (bone/liver/brain/adrenal)", "patient", None,
             (), (), "None available", "Metastasis prediction module — requires an appropriately labeled dataset.",
             "No public dataset used here has per-organ metastasis labels.", supported=False),
    TaskSpec("small_cell", "Small-cell lung cancer", "patient", None, (), (), "None available",
             "Histological subtype prediction is unavailable for small-cell lung cancer.",
             "No SCLC cases in the selected datasets.", supported=False),
]}


def risk_band(p: float, low: float = 0.33, high: float = 0.66) -> str:
    """Display-only bands; the cut-offs are arbitrary presentation choices, not clinical thresholds."""
    return "Low" if p < low else ("High" if p >= high else "Moderate")

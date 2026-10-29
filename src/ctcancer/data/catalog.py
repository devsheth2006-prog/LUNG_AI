"""Machine-readable dataset catalog: what each public dataset actually contains and which
tasks in this project it is (and is not) allowed to supervise.

This table drives the capability registry (which modules are 'supported by data') and is
rendered on the app's Home page and in the README. Facts verified against the TCIA collection
pages and the dataset descriptor papers (see README references); re-check before publication.
"""
from __future__ import annotations

DATASETS: dict[str, dict] = {
    "LIDC-IDRI": {
        "url": "https://www.cancerimagingarchive.net/collection/lidc-idri/",
        "imaging": "Thoracic CT (diagnostic + screening), DICOM; slice thickness 0.6–5 mm",
        "patients": "1,010 patients / 1,018 CT scans",
        "labels": ("Up to 4 radiologist reads per scan (blinded + unblinded); nodules ≥3 mm contoured with "
                   "per-reader malignancy rating 1–5 and 8 other semantic ratings; nodules <3 mm and "
                   "non-nodules ≥3 mm marked by position only"),
        "tasks": ["nodule_segmentation", "malignancy_suspicion", "radiomics", "pca_lda"],
        "limitations": ("Malignancy is a SUBJECTIVE radiologist suspicion score, not pathology (pathology "
                        "diagnosis exists for only a small subset of patients). No stage, no metastasis, no "
                        "histology, no clinical variables. Multiple readers per nodule must be grouped."),
        "license": "CC BY 3.0 (TCIA Data Usage Policy; cite Armato et al. 2011)",
    },
    "LUNA16": {
        "url": "https://luna16.grand-challenge.org/",
        "imaging": "888 LIDC-IDRI CT scans with slice thickness ≤2.5 mm, MetaImage (.mhd/.raw)",
        "patients": "888 scans (subset of LIDC-IDRI)",
        "labels": ("1,186 nodules ≥3 mm accepted by ≥3/4 radiologists (centre + diameter); candidate list "
                   "with nodule/non-nodule labels; automatically generated lung masks (seg-lungs-LUNA16)"),
        "tasks": ["lung_segmentation", "nodule_detection"],
        "limitations": ("Lung masks are algorithm-generated (not manual) — Dice against them measures agreement "
                        "with that algorithm. Same patients as LIDC-IDRI: never put a LIDC patient in the "
                        "LUNA16 test set and a LIDC training set."),
        "license": "CC BY 4.0 (challenge data, derived from LIDC-IDRI)",
    },
    "NSCLC-Radiomics (Lung1)": {
        "url": "https://www.cancerimagingarchive.net/collection/nsclc-radiomics/",
        "imaging": "Pre-treatment planning CT, DICOM; RTSTRUCT + DICOM-SEG with GTV-1",
        "patients": "422 NSCLC patients",
        "labels": ("Manual gross tumour volume (GTV) by a radiation oncologist; clinical CSV with age, sex, "
                   "clinical T/N/M stage, overall stage, histology (adenocarcinoma, squamous, large cell, NOS), "
                   "survival time and event"),
        "tasks": ["tumor_segmentation", "subtype", "t_category", "n_category", "survival_extension"],
        "limitations": ("All patients have cancer (no benign class). Staging is CLINICAL (imaging-based) not "
                        "pathological; very few M1 cases; histology missing/NOS for a sizeable fraction. "
                        "No small-cell lung cancer. Non-commercial use only."),
        "license": "CC BY-NC 3.0 (non-commercial; cite Aerts et al. 2014)",
    },
    "NSCLC-Radiogenomics": {
        "url": "https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/",
        "imaging": "CT (211 subjects) and PET/CT (201), DICOM; tumour segmentations for 144 subjects",
        "patients": "211 NSCLC subjects (R01 cohort 162 + AMC cohort 49)",
        "labels": ("Age, sex, smoking status, histology, PATHOLOGICAL T/N/M stage, recurrence and recurrence "
                   "location, survival, EGFR/KRAS mutation status, gene expression for a subset"),
        "tasks": ["tumor_segmentation", "subtype", "t_category", "n_category", "multimodal",
                  "distant_recurrence_extension"],
        "limitations": ("Small cohort (risk of overfitting; use grouped cross-validation, report CIs). "
                        "Pathological staging only for surgically resected patients (selection bias towards "
                        "early stage). No per-organ metastasis labels; 'recurrence location' is post-treatment "
                        "recurrence, not baseline metastasis."),
        "license": "CC BY 3.0 (cite Bakr et al. 2018)",
    },
    "MSD Task06 Lung": {
        "url": "http://medicaldecathlon.com/",
        "imaging": "CT, NIfTI",
        "patients": "63 labelled training cases (96 total)",
        "labels": "Primary lung tumour masks",
        "tasks": ["tumor_segmentation_external_validation"],
        "limitations": ("Cases originate from TCIA lung cancer collections — check for patient overlap with "
                        "NSCLC collections before using as an external test set."),
        "license": "CC BY-SA 4.0",
    },
}

# Things the specification asks for that NO dataset above can supervise.
UNSUPPORTED = {
    "small_cell_subtype": "No small-cell lung cancer cases in the NSCLC collections; SCLC is not predicted.",
    "organ_specific_metastasis": ("No public CT dataset here has per-organ (bone/liver/brain/adrenal) "
                                  "metastasis labels. Site-specific metastasis prediction requires an "
                                  "appropriately labelled dataset."),
    "official_tnm_stage": ("Stage labels support *model-estimated category probabilities* only; official "
                           "staging requires clinical work-up (PET, biopsy, MDT review)."),
    "lobe_location": "No lobe segmentation labels are used; locations are approximate (side/third/zone).",
}


def dataset_table() -> list[dict]:
    return [{"Dataset": k, "Tasks": ", ".join(v["tasks"]), "Labels": v["labels"], "Imaging": v["imaging"],
             "Patients": v["patients"], "Limitations": v["limitations"], "License": v["license"]}
            for k, v in DATASETS.items()]

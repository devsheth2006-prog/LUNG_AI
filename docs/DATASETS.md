# Datasets: what they contain, what they can supervise, how to get them

Facts below were checked against the TCIA collection pages and dataset descriptor papers
(Armato 2011; Setio 2017; Aerts 2014; Bakr 2018). Collection pages change — re-verify counts and
licences before you publish.

## 1. Summary table

| Dataset | Task(s) in this project | Labels actually available | Imaging | Patients | Key limitations |
|---|---|---|---|---|---|
| **LIDC-IDRI** | nodule segmentation, malignancy *suspicion*, radiomics, PCA/LDA study | ≤4 radiologist reads/scan; nodules ≥3 mm contoured; per-reader malignancy rating 1–5 + 8 semantic ratings | Thoracic CT, DICOM, 0.6–5 mm slices | 1,010 patients / 1,018 scans | Malignancy = radiologist opinion, **not pathology**; no stage, histology, metastasis or clinical data |
| **LUNA16** | lung segmentation, nodule detection (FP reduction) | 1,186 nodules accepted by ≥3/4 readers; labelled candidate list; algorithmic lung masks | 888 LIDC scans (≤2.5 mm), MetaImage | 888 scans | Same patients as LIDC; lung masks not manual |
| **NSCLC-Radiomics (Lung1)** | tumour segmentation, subtype, T, N, (M) | GTV contour (radiation oncologist); age, sex, **clinical** T/N/M, overall stage, histology, survival | Planning CT, DICOM + RTSTRUCT/SEG | 422 | All cancer (no benign); few M1; histology often NOS/missing; **CC BY-NC** (non-commercial) |
| **NSCLC-Radiogenomics** | tumour segmentation, subtype, T, N, multimodal, recurrence extension | Segmentation (144 subj.); age, sex, smoking, pack-years, histology, **pathological** T/N/M, recurrence + location, survival, EGFR/KRAS | CT (211) + PET/CT (201), DICOM | 211 | Small; surgically resected → early-stage bias; recurrence ≠ baseline metastasis |
| **MSD Task06 Lung** (optional) | external tumour-segmentation test | Tumour masks | CT, NIfTI | 63 labelled | Check overlap with TCIA NSCLC patients before using as external test |

## 2. What can and cannot be trained (and how the app words it)

| Module | Trainable? | Source | Output wording |
|---|---|---|---|
| Lung segmentation | ✅ | LUNA16 masks | lung mask (+ classical baseline when untrained) |
| Nodule detection | ✅ | LUNA16 candidates | probability that a candidate is a nodule |
| Lesion segmentation | ✅ | LIDC consensus + NSCLC GTV | lesion mask, volume, diameters |
| Malignancy | ⚠️ proxy | LIDC ratings (<3 vs >3) | *model-estimated malignancy probability (calibrated to radiologist suspicion)* |
| Subtype | ⚠️ NSCLC only | Lung1 + Radiogenomics histology | adenocarcinoma / squamous / other NSCLC probabilities |
| T category | ⚠️ | clinical + pathological T (mixed) | *model-estimated T-category probability distribution* |
| N involvement | ⚠️ | N0 vs N≥1 | *probability of regional nodal involvement* |
| M (distant) | ⚠️ rarely | very few M1 → refused below `risk.min_positive_labels` | refused / probability |
| Distant recurrence | extension | Radiogenomics recurrence location | prognostic research output |
| Small-cell subtype | ❌ | no SCLC cases | "Histological subtype prediction is unavailable…" |
| Organ-specific metastasis | ❌ | no labels | "Metastasis prediction module — requires an appropriately labeled dataset." |
| Official stage (I–IV) | ❌ | — | never computed |
| Lobe location | ❌ | no lobe masks | side / third / central-peripheral only |

## 3. Acquisition

1. **TCIA collections (LIDC-IDRI, NSCLC-Radiomics, NSCLC-Radiogenomics)** — open the collection
   page, add the CT (and SEG/RTSTRUCT) series to the cart, download the `.tcia` manifest and fetch
   it with the **NBIA Data Retriever** into `data/raw/<collection>/`. Download the clinical CSV from
   the same page. Accept the TCIA Data Usage Policy and cite the collection + paper.
   - LIDC-IDRI ≈ 125 GB; NSCLC-Radiomics ≈ 29 GB; Radiogenomics CT ≈ 20 GB.
   - For LIDC, install `pylidc` and create `~/.pylidcrc`:
     ```
     [dicom]
     path = /abs/path/to/data/raw/LIDC-IDRI
     warn = True
     ```
2. **LUNA16** — download `subset0–9`, `seg-lungs-LUNA16`, `annotations.csv`, `candidates_V2.csv`
   from the challenge site (Zenodo mirrors). For patient-level splits, derive
   `seriesuid_to_patient.csv` (columns `seriesuid,patient_id`) from the LIDC-IDRI DICOM headers.
3. **Lite profile on a laptop** — start with LUNA16 `subset0–1` and `--limit 100` on LIDC; the
   pipeline and splits are identical, only the amount of data changes.

## 4. Leakage traps specific to these datasets

- LIDC lists up to 4 annotations of the **same nodule**: annotations are clustered into one nodule
  before labelling; lesions, not annotations, are the samples — and splits are by **patient**.
- LUNA16 ⊂ LIDC: use the same patient split file for both if you combine them, or keep detection
  and malignancy experiments separate (as the scripts do).
- NSCLC-Radiomics vs Radiogenomics vs MSD: patient IDs are prefixed with the cohort name, and MSD
  must be overlap-checked before external testing.
- Imputation, scaling, PCA and LDA are fitted **inside** each training fold (sklearn Pipelines).

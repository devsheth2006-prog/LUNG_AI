# data/

No data ships with this repository. Download the public datasets yourself (see
[`docs/DATASETS.md`](../docs/DATASETS.md)) and place them as follows:

```
data/raw/
├── LIDC-IDRI/                      # NBIA download: LIDC-IDRI/<patient>/<study>/<series>/*.dcm
├── LUNA16/
│   ├── subset0 … subset9/          # *.mhd + *.raw
│   ├── seg-lungs-LUNA16/           # lung masks
│   ├── annotations.csv
│   ├── candidates_V2.csv
│   └── seriesuid_to_patient.csv    # optional, recommended (patient-level grouping)
├── NSCLC-Radiomics/
│   ├── LUNG1-001/ …                # CT + SEG/RTSTRUCT DICOM
│   └── NSCLC-Radiomics-Lung1.clinical.csv
└── NSCLC-Radiogenomics/
    ├── R01-001/ … AMC-001/ …
    └── NSCLCR01Radiogenomic_DATA_LABELS.csv
```

`training/prepare_data.py` writes cached patches to `data/processed/`, patient-level split
files to `data/splits/` and feature tables to `data/features/`. All of these are git-ignored.

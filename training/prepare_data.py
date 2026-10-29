"""Build cached training tables + patient-level splits from the raw public datasets.

Examples
--------
python training/prepare_data.py --dataset lidc                 # nodule patches + malignancy-suspicion labels
python training/prepare_data.py --dataset luna16_lung          # lung-mask slices for the U-Net
python training/prepare_data.py --dataset luna16_candidates    # detection FP-reduction patches
python training/prepare_data.py --dataset nsclc_radiomics      # GTV patches + clinical TNM/histology
python training/prepare_data.py --dataset nsclc_radiogenomics  # tumour patches + pathological TNM/clinical
Add `--limit 20` for a quick dry run on a few scans.
"""
import _bootstrap  # noqa: F401

from pathlib import Path

from ctcancer.config import cli_parser, load_config, set_seed
from ctcancer.data.splits import patient_level_split, save_split
from ctcancer.logging_utils import get_logger
from ctcancer.tracking import file_hash

log = get_logger("prepare_data")


def require(path: Path, what: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"\n[missing data] {what} not found at {path}\n"
            "Download it as described in docs/DATASETS.md and place it there (or override the path with "
            "--set datasets.<name>.root=...). No synthetic substitute is generated.\n")


def main():
    p = cli_parser(__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["lidc", "luna16_lung", "luna16_candidates", "nsclc_radiomics", "nsclc_radiogenomics"])
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    set_seed(cfg.seed)
    proc = cfg.path("processed_root")
    ds = cfg["datasets"]
    sp = tuple(cfg["preprocessing"]["target_spacing_mm"])
    fr = (cfg["splits"]["train"], cfg["splits"]["val"], cfg["splits"]["test"])

    if a.dataset == "lidc":
        from ctcancer.data.lidc import build_lidc_table

        require(Path(ds["lidc_idri"]["root"]), "LIDC-IDRI DICOM tree")
        out = proc / "lidc"
        df = build_lidc_table(out, sp, cfg["preprocessing"]["patch_size_vox"], ds["lidc_idri"]["min_readers"],
                              ds["lidc_idri"]["malignancy_rule"], a.limit,
                              dicom_root=Path(ds["lidc_idri"]["root"]))
        lab = df[df.label.notna()].copy()
        lab["label"] = lab.label.astype(int)
        split = patient_level_split(lab, "patient_id", "label", fr, cfg.seed)
        log.info("LIDC nodules: %d total, %d with a definite suspicion label (benign=%d, suspicious=%d)",
                 len(df), len(lab), int((lab.label == 0).sum()), int((lab.label == 1).sum()))
    elif a.dataset == "luna16_lung":
        from ctcancer.data.luna16 import build_lung_seg_slices

        require(Path(ds["luna16"]["root"]) / "seg-lungs-LUNA16", "LUNA16 lung masks")
        out = proc / "luna16_lung"
        df = build_lung_seg_slices(ds["luna16"]["root"], out, cfg["preprocessing"]["slice_size"], limit=a.limit)
        split = patient_level_split(df, "patient_id", None, fr, cfg.seed)
    elif a.dataset == "luna16_candidates":
        from ctcancer.data.luna16 import build_candidate_patches

        require(Path(ds["luna16"]["root"]) / "candidates_V2.csv", "LUNA16 candidates_V2.csv")
        out = proc / "luna16_candidates"
        df = build_candidate_patches(ds["luna16"]["root"], out, sp, 48, limit=a.limit)
        split = patient_level_split(df, "patient_id", "label", fr, cfg.seed)
    else:
        from ctcancer.data.clinical import parse_nsclc_radiogenomics, parse_nsclc_radiomics
        from ctcancer.data.nsclc import build_nsclc_table

        key = a.dataset
        require(Path(ds[key]["clinical_csv"]), f"{key} clinical CSV")
        labels = (parse_nsclc_radiomics if key == "nsclc_radiomics" else parse_nsclc_radiogenomics)(ds[key]["clinical_csv"])
        if a.limit:
            labels = labels.head(a.limit)
        out = proc / key
        df = build_nsclc_table(ds[key]["root"], labels, out, key, sp, 96)
        split = patient_level_split(df.assign(label=df.n_positive.fillna(0)), "patient_id", "label", fr, cfg.seed)
        log.info("label availability: %s", {c: int(df[c].notna().sum()) for c in
                                            ["histology", "t_category", "n_positive", "m_positive"] if c in df})
    split_path = Path(cfg.path("splits_dir")) / f"{a.dataset}.json"
    save_split(split, split_path, {"dataset": a.dataset, "seed": cfg.seed, "level": "patient",
                                   "table": str(out), "table_hash": file_hash(next(out.glob("*.csv")))})
    log.info("Saved patient-level split %s: %s", split_path, {k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()

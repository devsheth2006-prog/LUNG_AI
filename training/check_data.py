"""Check that the downloaded datasets are in the places the project expects.

python training/check_data.py

Prints OK / MISSING for every dataset folder and file, with a hint for anything missing.
It only looks at folders and file names; it never changes anything.
"""
import _bootstrap  # noqa: F401

import os
from pathlib import Path

from ctcancer.config import PROJECT_ROOT, cli_parser, load_config


def count(folder: Path, pattern: str) -> int:
    return sum(1 for _ in folder.rglob(pattern)) if folder.exists() else 0


def line(ok: bool, what: str, detail: str, hint: str = "") -> bool:
    print(f"  [{'OK' if ok else 'MISSING'}] {what}: {detail}")
    if not ok and hint:
        print(f"            -> {hint}")
    return ok


def main():
    a = cli_parser(__doc__).parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    ds = cfg["datasets"]
    root = lambda p: Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p  # noqa: E731
    ready = {}

    print("\nLUNA16  (lung masks + nodule detection)")
    luna = root(ds["luna16"]["root"])
    n = sum(count(luna / f"subset{i}", "*.mhd") for i in range(10))
    ok1 = line(n > 0, "CT scans (subset*/ *.mhd)", f"{n} scans", f"unzip subset0.zip into {luna}/subset0/")
    m = count(luna / "seg-lungs-LUNA16", "*.mhd")
    ok2 = line(m > 0, "lung masks (seg-lungs-LUNA16/)", f"{m} masks", f"unzip seg-lungs-LUNA16.zip into {luna}/")
    ok3 = line((luna / "candidates_V2.csv").exists(), "candidates_V2.csv", str(luna / "candidates_V2.csv"),
               "find candidates_V2.csv (it may be inside a zip) and put it directly in the LUNA16 folder")
    ready["luna16_lung"], ready["luna16_candidates"] = ok1 and ok2, ok1 and ok3

    print("\nLIDC-IDRI  (malignancy)")
    lidc = root(ds["lidc_idri"]["root"])
    pats = [p for p in lidc.glob("LIDC-IDRI-*") if p.is_dir()] if lidc.exists() else []
    ok1 = line(len(pats) > 0, "patient folders (LIDC-IDRI-0001 ...)", f"{len(pats)} patients",
               f"move the folders named LIDC-IDRI-xxxx directly into {lidc}/")
    rc = Path(os.path.expanduser("~/.pylidcrc"))
    rc_win = Path(os.path.expanduser("~/pylidc.conf"))
    has_rc = rc.exists() or rc_win.exists()
    ok2 = line(has_rc, "pylidc settings file", str(rc if rc.exists() else rc_win),
               "create it: see Step 8 (the pylidc settings note) in the dataset guide")
    if has_rc:
        text = (rc if rc.exists() else rc_win).read_text()
        line(str(lidc.resolve()) in text or str(lidc) in text, "settings point to the LIDC folder",
             "path found" if str(lidc) in text else "path not found in the settings file",
             f"the line should read:  path = {lidc.resolve()}")
    ready["lidc"] = ok1 and ok2

    for key, title, pid in (("nsclc_radiomics", "NSCLC-Radiomics  (tumours, subtype, T/N)", "LUNG1-"),
                            ("nsclc_radiogenomics", "NSCLC-Radiogenomics  (optional)", "R01-")):
        print(f"\n{title}")
        r = root(ds[key]["root"])
        pats = [p for p in r.iterdir() if p.is_dir() and p.name.startswith((pid, "AMC-"))] if r.exists() else []
        ok1 = line(len(pats) > 0, f"patient folders ({pid}xxx ...)", f"{len(pats)} patients",
                   f"move the folders named {pid}xxx directly into {r}/ (not inside a manifest-... folder)")
        csv = root(ds[key]["clinical_csv"])
        ok2 = line(csv.exists(), "clinical CSV", csv.name, f"download it and rename it to exactly {csv.name}, "
                   f"then put it in {r}/")
        ready[key] = ok1 and ok2

    print("\nYou can now run:")
    any_ready = False
    for k, v in ready.items():
        if v:
            any_ready = True
            print(f"  python training/prepare_data.py --dataset {k}")
    if not any_ready:
        print("  (nothing yet - fix the MISSING lines above first)")
    print()


if __name__ == "__main__":
    main()

"""Run the full analysis on one scan from the command line.

python inference/inference.py --input path/to/scan.nii.gz --report outputs/reports
python inference/inference.py --input dicom_dir/ --age 67 --sex male --json-only
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

from ctcancer.config import cli_parser, load_config  # noqa: E402
from ctcancer.inference.pipeline import AnalysisPipeline  # noqa: E402
from ctcancer.reporting.report import build_report  # noqa: E402


def main():
    p = cli_parser(__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--report", default="outputs/reports")
    p.add_argument("--age", type=float, default=None)
    p.add_argument("--sex", choices=["male", "female"], default=None)
    p.add_argument("--json-only", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    clinical = {}
    if a.age is not None:
        clinical["age"] = a.age
    if a.sex:
        clinical["sex_male"] = 1.0 if a.sex == "male" else 0.0
    res, arrays = AnalysisPipeline(cfg).run(a.input, clinical)
    if not res.ok:
        print(f"Analysis failed: {res.error}")
        sys.exit(1)
    out = Path(a.report)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{res.scan['scan_id']}.json").write_text(json.dumps(res.to_dict(), indent=2))
    if not a.json_only:
        paths = build_report(res, arrays, out)
        print("Report:", ", ".join(str(p) for p in paths.values()))
    print(json.dumps({k: res.to_dict()[k] for k in ("scan", "lung", "index_lesion")}, indent=2))


if __name__ == "__main__":
    main()

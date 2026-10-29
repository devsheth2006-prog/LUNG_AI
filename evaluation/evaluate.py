"""Summarise every trained model from its model card (and regenerate figures).

python evaluation/evaluate.py              # prints a table, writes outputs/model_summary.{md,json}

All numbers come from the held-out test evaluations performed by the training scripts. This
script never computes or invents metrics for models that have not been trained.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

from ctcancer.config import PROJECT_ROOT, cli_parser, load_config  # noqa: E402
from ctcancer.inference.registry import all_statuses  # noqa: E402

KEYS = ["roc_auc", "pr_auc", "recall_sensitivity", "sensitivity", "specificity", "f1", "brier", "ece", "dice", "iou",
        "hd95_mm", "balanced_accuracy", "macro_f1", "macro_roc_auc_ovr"]


def main():
    p = cli_parser(__doc__)
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    rows = []
    for task, st in all_statuses(cfg.path("models_dir")).items():
        m = (st.card or {}).get("test_metrics", {}) if st.state == "evaluated" else {}
        rows.append({"task": task, "status": st.badge, "model": (st.card or {}).get("architecture", "—"),
                     **{k: m[k] for k in KEYS if k in m}, "n_test_patients": m.get("n_test_patients")})
    lines = ["| task | status | model | key metrics |", "|---|---|---|---|"]
    for r in rows:
        km = ", ".join(f"{k}={r[k]:.3f}" for k in KEYS if isinstance(r.get(k), float))
        lines.append(f"| {r['task']} | {r['status']} | {r['model']} | {km or '—'} |")
    out = PROJECT_ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "model_summary.md").write_text("\n".join(lines) + "\n")
    (out / "model_summary.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

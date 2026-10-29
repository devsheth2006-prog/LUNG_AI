"""Train spread-risk estimators: regional nodal involvement (N0 vs N+), distant metastasis
(M1, only if enough positives) and distant recurrence (research extension, Radiogenomics).

python training/train_metastasis_model.py

Organ-specific metastasis (bone/liver/brain/adrenal) is NOT trained: no dataset used here has
those labels. The registry reports it as "Metastasis prediction module — requires an
appropriately labeled dataset."
"""
import _bootstrap  # noqa: F401

from _patient_tasks import run_tasks
from ctcancer.config import cli_parser, load_config, set_seed

TASKS = ["n_involvement", "m_distant", "distant_recurrence"]


def main():
    p = cli_parser(__doc__)
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    set_seed(cfg.seed)
    run_tasks(cfg, a.tasks)


if __name__ == "__main__":
    main()

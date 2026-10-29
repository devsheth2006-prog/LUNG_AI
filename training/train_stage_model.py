"""Train subtype and T-category estimators on NSCLC cohorts (patient level).

python training/train_stage_model.py                    # subtype + t_category
python training/train_stage_model.py --tasks subtype

Compares logistic regression / random forest / gradient boosting over radiomic, CNN, PCA/LDA
and clinical feature sets with grouped CV; refuses tasks with too few labels per class.
Outputs are *model-estimated category probabilities* — never an official stage.
"""
import _bootstrap  # noqa: F401

from _patient_tasks import run_tasks
from ctcancer.config import cli_parser, load_config, set_seed


def main():
    p = cli_parser(__doc__)
    p.add_argument("--tasks", nargs="+", default=["subtype", "t_category"], choices=["subtype", "t_category"])
    a = p.parse_args()
    cfg = load_config(a.config, a.set, a.profile)
    set_seed(cfg.seed)
    run_tasks(cfg, a.tasks)


if __name__ == "__main__":
    main()

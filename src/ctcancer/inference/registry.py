"""Model registry: which models exist, what they were trained/evaluated on, and their status.

Every training script writes `models/<task>/model_card.json`. The UI only ever displays
metrics read from these cards (computed by the evaluation code on the held-out test set). If
a card is missing the UI shows "Model not trained" — no placeholder numbers exist anywhere.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..risk.tasks import TASKS

MODEL_FILES = {
    "lung_segmentation": "model.pt", "tumor_segmentation": "model.pt", "nodule_detection": "model.pt",
    "malignancy": "model.pt", "subtype": "model.joblib", "t_category": "model.joblib",
    "n_involvement": "model.joblib", "m_distant": "model.joblib", "distant_recurrence": "model.joblib",
}


@dataclass
class TaskStatus:
    task: str
    state: str          # evaluated | trained_unevaluated | not_trained | refused | unsupported
    message: str
    card: dict | None = None

    @property
    def usable(self) -> bool:
        return self.state in ("evaluated", "trained_unevaluated")

    @property
    def badge(self) -> str:
        return {"evaluated": "Trained & evaluated", "trained_unevaluated": "Trained — not yet evaluated",
                "not_trained": "Model not trained", "refused": "Insufficient labels",
                "unsupported": "Not supported by available data"}[self.state]


def models_dir(root: str | Path | None = None) -> Path:
    """Explicit root, else `paths.models_dir` from the active config (CTC_CONFIG aware)."""
    if root:
        return Path(root)
    from ..config import load_config

    return load_config().path("models_dir")


def write_card(task: str, card: dict, root: str | Path | None = None) -> Path:
    d = models_dir(root) / task
    d.mkdir(parents=True, exist_ok=True)
    card = {"task": task, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **card}
    spec = TASKS.get(task)
    if spec:
        card.setdefault("label_kind", spec.label_kind)
        card.setdefault("caveat", spec.caveat)
    (d / "model_card.json").write_text(json.dumps(card, indent=2, default=float))
    return d / "model_card.json"


def read_card(task: str, root: str | Path | None = None) -> dict | None:
    f = models_dir(root) / task / "model_card.json"
    return json.loads(f.read_text()) if f.exists() else None


def task_status(task: str, root: str | Path | None = None) -> TaskStatus:
    spec = TASKS.get(task)
    if spec is not None and not spec.supported:
        return TaskStatus(task, "unsupported", spec.output_wording)
    card = read_card(task, root)
    if card is None:
        return TaskStatus(task, "not_trained", "Model not trained. Run the training script for this task "
                          "(see README › Training).")
    if card.get("refused"):
        return TaskStatus(task, "refused", card.get("reason", "Training refused: insufficient labels."), card)
    if not (models_dir(root) / task / MODEL_FILES.get(task, "model.pt")).exists():
        return TaskStatus(task, "not_trained", "Model card found but weights are missing.", card)
    if not card.get("test_metrics"):
        return TaskStatus(task, "trained_unevaluated", "Trained, but no held-out test evaluation recorded.", card)
    return TaskStatus(task, "evaluated", "Evaluated on a held-out patient-level test set.", card)


def all_statuses(root: str | Path | None = None) -> dict[str, TaskStatus]:
    return {k: task_status(k, root) for k in TASKS}

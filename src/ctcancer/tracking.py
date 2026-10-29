"""Experiment tracking: MLflow when installed, otherwise an equivalent JSON-lines run log.

Both record params, per-epoch metrics, final metrics, model version, dataset version
(hash of the split file) and experiment name, so runs stay comparable either way.
"""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path


def file_hash(path: str | Path) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else "missing"


class Tracker:
    def __init__(self, experiment: str, run_name: str, root: str | Path, backend: str = "auto"):
        self.experiment, self.run_name, self.root = experiment, run_name, Path(root)
        self.backend = backend
        self._mlflow = None
        if backend in ("auto", "mlflow"):
            try:
                import mlflow

                mlflow.set_tracking_uri(f"file:{self.root.resolve()}")
                mlflow.set_experiment(experiment)
                self._mlflow = mlflow
            except ImportError:
                if backend == "mlflow":
                    raise
        self.record = {"experiment": experiment, "run": run_name, "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "params": {}, "history": [], "metrics": {}, "tags": {}}

    @contextmanager
    def run(self):
        if self._mlflow:
            with self._mlflow.start_run(run_name=self.run_name):
                yield self
        else:
            yield self
        self.record["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "runs.jsonl", "a") as f:
            f.write(json.dumps(self.record, default=float) + "\n")

    def log_params(self, params: dict):
        flat = _flatten(params)
        self.record["params"].update(flat)
        if self._mlflow:
            self._mlflow.log_params({k: str(v)[:250] for k, v in flat.items()})

    def log_metrics(self, metrics: dict, step: int | None = None):
        num = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if step is None:
            self.record["metrics"].update(num)
        else:
            self.record["history"].append({"step": step, **num})
        if self._mlflow:
            self._mlflow.log_metrics(num, step=step)

    def set_tags(self, tags: dict):
        self.record["tags"].update(tags)
        if self._mlflow:
            self._mlflow.set_tags(tags)

    def log_artifact(self, path: str | Path):
        if self._mlflow and Path(path).exists():
            self._mlflow.log_artifact(str(path))


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out

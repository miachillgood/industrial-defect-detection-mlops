"""Thin MLflow wrapper.

Experiment tracking is an *addition* to the detection core, not part of it.
It is deliberately fail-soft: if MLflow is missing or the tracking store is
unreachable, evaluation still runs and still writes ``results.json``.

Tracking URI resolution order:
1. ``MLFLOW_TRACKING_URI`` environment variable
2. ``sqlite:///mlflow.db`` (local file-backed database, no server needed)

MLflow 3.15 put the plain-file backend (``file:./mlruns``) into maintenance mode
and now *raises* on it unless ``MLFLOW_ALLOW_FILE_STORE=true`` is set, so SQLite
is the default here. It also unlocks the features the file store never had
(model registry, metric search).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


class MlflowTracker:
    """Context-free tracker; call ``end()`` when finished."""

    def __init__(self, experiment: str = "spade-mvtec", run_name: str | None = None) -> None:
        import mlflow  # imported lazily so the core package has no hard dependency

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment)
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow rejects params > 500 chars and non-scalar values.
        clean = {k: (str(v)[:500]) for k, v in params.items() if v is not None}
        self._mlflow.log_params(clean)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {
            k.replace("/", "_"): float(v)
            for k, v in metrics.items()
            if v is not None and isinstance(v, (int, float))
        }
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifacts(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            self._mlflow.log_artifacts(str(path))
        elif path.exists():
            self._mlflow.log_artifact(str(path))

    def set_tags(self, tags: dict[str, str]) -> None:
        self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def end(self, status: str = "FINISHED") -> None:
        self._mlflow.end_run(status=status)


def log_existing_run(results_json: str | Path, experiment: str = "spade-mvtec") -> str | None:
    """Backfill MLflow from a ``results.json`` produced by an earlier offline run."""
    import json

    results_json = Path(results_json)
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    tracker = MlflowTracker(experiment=experiment, run_name=payload.get("run_name"))
    tracker.log_params(payload.get("config", {}))
    summary = payload.get("summary", {})
    tracker.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
    for row in payload.get("per_category", []):
        tracker.log_metrics(
            {
                f"image_rocauc_{row['category']}": row["image_rocauc"],
                f"pixel_rocauc_{row['category']}": row["pixel_rocauc"],
            }
        )
    tracker.log_artifacts(results_json.parent)
    run_id = tracker.run_id
    tracker.end()
    return run_id

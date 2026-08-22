"""MLflow wrapper.

The property this module advertises everywhere -- "a tracking failure never
breaks a run" -- had no test at all. These cover both the happy path against a
real temporary SQLite store and the fail-soft contract.
"""

from __future__ import annotations

import json

import pytest

from mlops.tracking import DEFAULT_TRACKING_URI, MlflowTracker, log_existing_run, tracking_uri


@pytest.fixture
def sqlite_store(tmp_path, monkeypatch):
    """Point MLflow at a throwaway SQLite file and cwd, so nothing leaks."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def test_default_uri_is_a_database_not_the_file_store():
    """MLflow 3.15 raises on `file:./mlruns` unless MLFLOW_ALLOW_FILE_STORE=true."""
    assert DEFAULT_TRACKING_URI.startswith("sqlite:")


def test_tracking_uri_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.invalid:5000")
    assert tracking_uri() == "http://example.invalid:5000"


def test_tracking_uri_falls_back_to_the_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking_uri() == DEFAULT_TRACKING_URI


def test_roundtrip_params_metrics_and_artifacts(sqlite_store, tmp_path):
    import mlflow

    artifact = tmp_path / "run" / "results.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("# report", encoding="utf-8")

    tracker = MlflowTracker(experiment="unit-test", run_name="r1")
    tracker.log_params({"top_k": 5, "backbone": "wide_resnet50_2"})
    tracker.log_metrics({"mean_image_rocauc": 85.41, "mean_pixel_rocauc": 96.44})
    tracker.log_artifacts(artifact.parent)
    run_id = tracker.run_id
    tracker.end()

    mlflow.set_tracking_uri(sqlite_store)
    run = mlflow.get_run(run_id)
    assert run.data.params["top_k"] == "5"
    assert run.data.metrics["mean_image_rocauc"] == pytest.approx(85.41)
    assert run.info.status == "FINISHED"


def test_log_params_drops_none_and_truncates_long_values(sqlite_store):
    import mlflow

    tracker = MlflowTracker(experiment="unit-test", run_name="r2")
    tracker.log_params({"kept": "a", "dropped": None, "long": "x" * 900})
    run_id = tracker.run_id
    tracker.end()

    mlflow.set_tracking_uri(sqlite_store)
    params = mlflow.get_run(run_id).data.params
    assert "dropped" not in params, "None params would make MLflow reject the whole call"
    assert len(params["long"]) == 500, "MLflow rejects params longer than 500 chars"


def test_log_metrics_skips_none_and_sanitises_slashes(sqlite_store):
    """Per-category keys arrive as `image_rocauc/bottle`; MLflow dislikes slashes."""
    import mlflow

    tracker = MlflowTracker(experiment="unit-test", run_name="r3")
    tracker.log_metrics({"image_rocauc/bottle": 97.22, "pixel_pro/bottle": None})
    run_id = tracker.run_id
    tracker.end()

    mlflow.set_tracking_uri(sqlite_store)
    metrics = mlflow.get_run(run_id).data.metrics
    assert "image_rocauc_bottle" in metrics
    assert not any("/" in k for k in metrics)
    assert "pixel_pro_bottle" not in metrics, "a None metric (PRO off) must be skipped"


def test_log_metrics_with_nothing_loggable_is_a_no_op(sqlite_store):
    tracker = MlflowTracker(experiment="unit-test", run_name="r4")
    tracker.log_metrics({"a": None, "b": "not a number"})  # must not raise
    tracker.end()


def test_set_tags_and_failed_status(sqlite_store):
    import mlflow

    tracker = MlflowTracker(experiment="unit-test", run_name="r5")
    tracker.set_tags({"stage": "evaluate", "categories": 15})
    run_id = tracker.run_id
    tracker.end(status="FAILED")

    mlflow.set_tracking_uri(sqlite_store)
    run = mlflow.get_run(run_id)
    assert run.data.tags["stage"] == "evaluate"
    assert run.data.tags["categories"] == "15"
    assert run.info.status == "FAILED"


def test_log_artifacts_accepts_a_single_file(sqlite_store, tmp_path):
    single = tmp_path / "metrics.json"
    single.write_text("{}", encoding="utf-8")
    tracker = MlflowTracker(experiment="unit-test", run_name="r6")
    tracker.log_artifacts(single)  # must not raise on a file rather than a dir
    tracker.end()


def test_log_artifacts_ignores_a_missing_path(sqlite_store, tmp_path):
    tracker = MlflowTracker(experiment="unit-test", run_name="r7")
    tracker.log_artifacts(tmp_path / "does-not-exist")  # must not raise
    tracker.end()


def test_log_existing_run_backfills_an_offline_run(sqlite_store, tmp_path):
    """The recovery path used after MLflow was unavailable during a real run."""
    import mlflow

    run_dir = tmp_path / "full-mvtec-k5"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "run_name": "full-mvtec-k5",
                "config": {"top_k": 5, "backbone": "wide_resnet50_2"},
                "summary": {
                    "mean_image_rocauc": 85.41,
                    "mean_pixel_rocauc": 96.44,
                    "n_categories": 15,
                },
                "per_category": [
                    {"category": "bottle", "image_rocauc": 97.22, "pixel_rocauc": 97.01},
                    {"category": "grid", "image_rocauc": 47.28, "pixel_rocauc": 98.35},
                ],
            }
        ),
        encoding="utf-8",
    )

    run_id = log_existing_run(run_dir / "results.json", experiment="backfill-test")
    assert run_id

    mlflow.set_tracking_uri(sqlite_store)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["mean_image_rocauc"] == pytest.approx(85.41)
    assert run.data.metrics["image_rocauc_bottle"] == pytest.approx(97.22)
    assert run.data.metrics["pixel_rocauc_grid"] == pytest.approx(98.35)
    assert run.info.run_name == "full-mvtec-k5"


def test_constructing_against_an_unreachable_store_raises(monkeypatch, tmp_path):
    """Not fail-soft by itself -- the *caller* is.

    ``spade.evaluate.run`` wraps construction in try/except precisely because
    this raises. See tests/test_evaluate.py::test_run_survives_a_broken_tracker.
    """
    from mlflow.exceptions import MlflowException

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    with pytest.raises(MlflowException, match="maintenance mode"):
        MlflowTracker(experiment="unit-test", run_name="boom")

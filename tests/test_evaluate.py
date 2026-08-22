"""End-to-end orchestration.

`spade.evaluate.run()` was the least-tested module in the project (29 % line
coverage) despite being what `make eval` and the DVC pipeline actually call.
These drive it over a synthetic MVTec-shaped tree so the whole path -- dataset
indexing, feature extraction, scoring, metrics, figures, report generation --
is exercised without the 5 GB dataset.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from spade.config import SpadeConfig
from spade.evaluate import _results_markdown, run

pytestmark = pytest.mark.needs_backbone

SIZE = 256


def _write(path, rng, defect=False):
    """A 'normal' image is smooth noise; a defective one gets a bright patch."""
    arr = (rng.normal(120, 8, (SIZE, SIZE, 3))).clip(0, 255).astype(np.uint8)
    if defect:
        arr[90:150, 90:150] = 245
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _write_mask(path):
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[90:150, 90:150] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    """A minimal MVTec AD tree. `bottle` is used because MVTecDataset validates
    the category name against the official 15."""
    root = tmp_path_factory.mktemp("mvtec") / "mvtec_anomaly_detection"
    rng = np.random.default_rng(0)
    cat = root / "bottle"

    for i in range(6):
        _write(cat / "train" / "good" / f"{i:03d}.png", rng)
    for i in range(3):
        _write(cat / "test" / "good" / f"{i:03d}.png", rng)
    for i in range(3):
        _write(cat / "test" / "broken" / f"{i:03d}.png", rng, defect=True)
        _write_mask(cat / "ground_truth" / "broken" / f"{i:03d}_mask.png")
    return root


def _cfg(synthetic_root, tmp_path, **kw):
    return SpadeConfig(
        data_root=str(synthetic_root),
        device="cpu",
        batch_size=4,
        num_workers=0,
        categories=("bottle",),
        output_dir=str(tmp_path / "runs"),
        save_visualizations=kw.pop("save_visualizations", 2),
        **kw,
    )


def test_run_end_to_end_writes_every_declared_output(synthetic_root, tmp_path):
    payload = run(_cfg(synthetic_root, tmp_path), run_name="t", use_mlflow=False)
    out = tmp_path / "runs" / "t"

    for name in ("results.json", "results.md", "metrics.json", "config.json",
                 "roc_curves.json", "roc_curve.png"):
        assert (out / name).is_file(), f"{name} is declared as a DVC output but was not written"
    assert (out / "images").is_dir()
    assert len(list((out / "images").glob("bottle_*.png"))) == 2

    assert payload["summary"]["n_categories"] == 1
    row = payload["per_category"][0]
    assert row["category"] == "bottle"
    assert row["n_train"] == 6 and row["n_test"] == 6 and row["n_anomalous"] == 3
    for key in ("image_rocauc", "pixel_rocauc"):
        assert 0.0 <= row[key] <= 100.0


def test_run_separates_the_injected_defect(synthetic_root, tmp_path):
    """A bright patch on an otherwise smooth image should be trivially separable."""
    payload = run(_cfg(synthetic_root, tmp_path, save_visualizations=0), run_name="sep",
                  use_mlflow=False)
    row = payload["per_category"][0]
    assert row["image_rocauc"] > 80.0
    assert row["pixel_rocauc"] > 80.0


def test_metrics_json_holds_only_scalars(synthetic_root, tmp_path):
    """`dvc metrics show` cannot render nested structures."""
    run(_cfg(synthetic_root, tmp_path, save_visualizations=0), run_name="m", use_mlflow=False)
    metrics = json.loads((tmp_path / "runs" / "m" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics
    assert all(isinstance(v, (int, float)) for v in metrics.values())
    assert "mean_image_rocauc" in metrics and "mean_pixel_rocauc" in metrics


def test_pro_is_computed_by_default_and_can_be_disabled(synthetic_root, tmp_path):
    with_pro = run(_cfg(synthetic_root, tmp_path, save_visualizations=0), run_name="p1",
                   use_mlflow=False)
    assert with_pro["summary"].get("mean_pixel_pro") is not None
    assert with_pro["per_category"][0]["pixel_pro"] is not None

    without = run(_cfg(synthetic_root, tmp_path, save_visualizations=0), run_name="p2",
                  use_mlflow=False, compute_pro=False)
    assert without["summary"].get("mean_pixel_pro") is None
    assert without["per_category"][0]["pixel_pro"] is None


def test_run_survives_a_broken_tracker(synthetic_root, tmp_path, monkeypatch, capsys):
    """The fail-soft contract this project advertises everywhere.

    Tracking is observation, not a runtime dependency: if MLflow cannot be
    constructed, the evaluation must still finish and write results.json. This
    is not hypothetical -- MLflow 3.15 started raising on the file store and a
    real 15-category run survived it exactly this way.
    """
    import mlops.tracking as tracking

    class Exploding:
        def __init__(self, *a, **kw):
            raise RuntimeError("tracking backend is in maintenance mode")

    monkeypatch.setattr(tracking, "MlflowTracker", Exploding)

    payload = run(_cfg(synthetic_root, tmp_path, save_visualizations=0), run_name="soft",
                  use_mlflow=True)

    assert (tmp_path / "runs" / "soft" / "results.json").is_file()
    assert payload["summary"]["mean_image_rocauc"] is not None
    assert "MLflow disabled" in capsys.readouterr().out


def test_unknown_category_is_rejected_before_any_work(synthetic_root, tmp_path):
    cfg = _cfg(synthetic_root, tmp_path)
    cfg.categories = ("bottle", "not_a_category")
    with pytest.raises(ValueError, match="not_a_category"):
        run(cfg, run_name="bad", use_mlflow=False)


def test_missing_dataset_fails_with_a_pointed_message(tmp_path):
    cfg = SpadeConfig(data_root=str(tmp_path / "nope"), device="cpu",
                      categories=("bottle",), output_dir=str(tmp_path / "runs"))
    with pytest.raises(FileNotFoundError, match="bottle/train/good"):
        run(cfg, run_name="x", use_mlflow=False)


def test_config_roundtrips_through_the_run_directory(synthetic_root, tmp_path):
    run(_cfg(synthetic_root, tmp_path, save_visualizations=0), run_name="c", use_mlflow=False)
    restored = SpadeConfig.load(tmp_path / "runs" / "c" / "config.json")
    assert restored.top_k == 5
    assert restored.categories == ("bottle",)
    assert restored.layers == ("layer1", "layer2", "layer3")


# --------------------------------------------------------------- report text
@pytest.mark.parametrize("with_pro", [True, False])
def test_results_markdown_shape(with_pro):
    rows = [{"category": "bottle", "image_rocauc": 97.22, "pixel_rocauc": 97.01,
             **({"pixel_pro": 93.33} if with_pro else {})}]
    summary = {"mean_image_rocauc": 97.22, "mean_pixel_rocauc": 97.01,
               **({"mean_pixel_pro": 93.33} if with_pro else {})}
    md = _results_markdown(rows, summary, SpadeConfig())

    assert "Image-level ROC-AUC" in md and "Pixel-level ROC-AUC" in md
    assert "IMAGENET1K_V1" in md
    assert "LANCZOS" in md
    assert ("## PRO (%)" in md) is with_pro
    if with_pro:
        assert "93.33" in md
        assert "no reference column" in md, "PRO has no public baseline to compare against"

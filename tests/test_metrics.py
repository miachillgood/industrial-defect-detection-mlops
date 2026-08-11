from __future__ import annotations

import numpy as np
import pytest

from spade.data import CLASS_NAMES
from spade.metrics import (
    PAPER_PIXEL_ROCAUC,
    REFERENCE_IMAGE_ROCAUC,
    REFERENCE_MEAN_IMAGE_ROCAUC,
    REFERENCE_MEAN_PIXEL_ROCAUC,
    REFERENCE_PIXEL_ROCAUC,
    image_level_metrics,
    per_region_overlap,
    pixel_level_metrics,
    summarize,
)


def test_image_level_perfect_separation():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])
    auc, curve = image_level_metrics(labels, scores)
    assert auc == pytest.approx(1.0)
    assert len(curve["fpr"]) == len(curve["tpr"])


def test_image_level_random_is_half():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    auc, _ = image_level_metrics(labels, scores)
    assert auc == pytest.approx(0.5)


def test_pixel_level_perfect_localization():
    masks = np.zeros((2, 1, 8, 8), dtype=np.uint8)
    masks[:, :, 2:5, 2:5] = 1
    maps = np.zeros((2, 8, 8), dtype=np.float32)
    maps[:, 2:5, 2:5] = 5.0
    auc, curve, threshold, f1 = pixel_level_metrics(masks, maps)
    assert auc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert 0.0 < threshold <= 5.0


def test_pro_rewards_small_regions_equally():
    """Pixel ROC-AUC is dominated by the big blob; PRO is not."""
    masks = np.zeros((1, 1, 32, 32), dtype=np.uint8)
    masks[0, 0, 0:16, 0:16] = 1  # large region
    masks[0, 0, 28:30, 28:30] = 1  # small region

    maps = np.zeros((1, 32, 32), dtype=np.float32)
    maps[0, 0:16, 0:16] = 9.0  # detected
    maps[0, 28:30, 28:30] = 0.0  # missed

    pixel_auc, *_ = pixel_level_metrics(masks, maps)
    pro = per_region_overlap(masks, maps)
    assert pixel_auc > 0.9
    assert pro < pixel_auc  # missing an entire region costs PRO much more


def test_pro_returns_nan_without_regions():
    masks = np.zeros((2, 1, 8, 8), dtype=np.uint8)
    maps = np.random.rand(2, 8, 8).astype(np.float32)
    assert np.isnan(per_region_overlap(masks, maps))


def test_summarize_averages_over_categories():
    rows = [
        {"category": "a", "image_rocauc": 90.0, "pixel_rocauc": 95.0, "pixel_pro": 80.0},
        {"category": "b", "image_rocauc": 80.0, "pixel_rocauc": 97.0, "pixel_pro": 90.0},
    ]
    s = summarize(rows)
    assert s["n_categories"] == 2
    assert s["mean_image_rocauc"] == pytest.approx(85.0)
    assert s["mean_pixel_rocauc"] == pytest.approx(96.0)
    assert s["mean_pixel_pro"] == pytest.approx(85.0)


def test_reference_tables_cover_all_categories_and_match_headline_means():
    """Guards the public baseline numbers this project is checked against."""
    for table in (REFERENCE_IMAGE_ROCAUC, REFERENCE_PIXEL_ROCAUC, PAPER_PIXEL_ROCAUC):
        assert set(table) == set(CLASS_NAMES)

    assert np.mean(list(REFERENCE_IMAGE_ROCAUC.values())) == pytest.approx(
        REFERENCE_MEAN_IMAGE_ROCAUC, abs=0.06
    )
    assert np.mean(list(REFERENCE_PIXEL_ROCAUC.values())) == pytest.approx(
        REFERENCE_MEAN_PIXEL_ROCAUC, abs=0.06
    )

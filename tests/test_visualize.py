"""Figure helpers. These have no heavy dependencies beyond matplotlib."""

from __future__ import annotations

import numpy as np
import pytest

from spade.visualize import plot_roc_curves, render_heatmap_overlay, save_localization_panels


def test_overlay_shape_and_dtype():
    """Regression guard: ``matplotlib.cm.get_cmap`` was removed in matplotlib 3.9.

    The overlay is what both the API and the review tool render, so a colormap
    lookup that only works on old matplotlib breaks two surfaces at once.
    """
    img = np.full((32, 32, 3), 40, dtype=np.uint8)
    score_map = np.random.rand(32, 32).astype(np.float32)
    out = render_heatmap_overlay(img, score_map)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_overlay_handles_a_constant_score_map():
    """A perfectly uniform map must not divide by zero."""
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = render_heatmap_overlay(img, np.full((8, 8), 3.0, dtype=np.float32))
    assert np.isfinite(out).all()


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_overlay_alpha_endpoints(alpha):
    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    score_map = np.zeros((8, 8), dtype=np.float32)
    out = render_heatmap_overlay(img, score_map, alpha=alpha)
    if alpha == 0.0:
        assert (out == 200).all()
    else:
        assert not (out == 200).all()


def test_plot_roc_curves_writes_a_png(tmp_path):
    rows = [
        {"category": "bottle", "image_rocauc": 97.2, "pixel_rocauc": 97.0},
        {"category": "cable", "image_rocauc": 84.8, "pixel_rocauc": 92.3},
    ]
    curves = {
        c["category"]: {
            "image": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.9, 1.0]},
            "pixel": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.95, 1.0]},
        }
        for c in rows
    }
    out = plot_roc_curves(rows, curves, tmp_path / "roc.png")
    assert out.exists() and out.stat().st_size > 0


def test_localization_panels_prefer_anomalous_samples(tmp_path):
    n = 6
    images = np.random.rand(n, 3, 32, 32).astype(np.float32)
    masks = np.zeros((n, 1, 32, 32), dtype=np.uint8)
    masks[3:, :, 8:16, 8:16] = 1
    score_maps = np.random.rand(n, 32, 32).astype(np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1])

    written = save_localization_panels(
        images, masks, score_maps, labels, threshold=0.5,
        out_dir=tmp_path, category="demo", max_panels=2,
    )
    assert len(written) == 2
    assert all(p.exists() for p in written)

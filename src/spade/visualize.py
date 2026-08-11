"""Figures: ROC curves and per-image localisation panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import denormalize  # noqa: E402


def plot_roc_curves(rows: list[dict], curves: dict, out_path: str | Path) -> Path:
    """Two-panel ROC figure, one curve per category -- same layout as the reference."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for row in rows:
        cat = row["category"]
        c = curves.get(cat, {})
        if c.get("image"):
            ax[0].plot(c["image"]["fpr"], c["image"]["tpr"],
                       label=f"{cat} ROCAUC: {row['image_rocauc'] / 100:.3f}")
        if c.get("pixel"):
            ax[1].plot(c["pixel"]["fpr"], c["pixel"]["tpr"],
                       label=f"{cat} ROCAUC: {row['pixel_rocauc'] / 100:.3f}")

    mean_img = np.mean([r["image_rocauc"] for r in rows]) / 100
    mean_pix = np.mean([r["pixel_rocauc"] for r in rows]) / 100
    ax[0].title.set_text(f"Average image ROCAUC: {mean_img:.3f}")
    ax[1].title.set_text(f"Average pixel ROCAUC: {mean_pix:.3f}")
    for a in ax:
        a.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
        a.set_xlabel("False positive rate")
        a.set_ylabel("True positive rate")
        a.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def save_localization_panels(
    images: np.ndarray,
    masks: np.ndarray,
    score_maps: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    out_dir: str | Path,
    category: str,
    max_panels: int = 5,
) -> list[Path]:
    """Image / ground truth / predicted mask / masked-out prediction, per sample.

    Anomalous samples are preferred -- a panel of a defect-free image says nothing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order = [i for i, y in enumerate(labels) if y == 1][:max_panels]
    if len(order) < max_panels:
        order += [i for i, y in enumerate(labels) if y == 0][: max_panels - len(order)]

    written: list[Path] = []
    for n, idx in enumerate(order):
        img = denormalize(images[idx])
        gt = np.asarray(masks[idx]).squeeze()
        heat = score_maps[idx]
        pred = (heat > threshold).astype(np.uint8)

        masked = img.copy()
        masked[pred == 0] = 0

        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.90, wspace=0.05)
        for a in axes:
            a.axes.xaxis.set_visible(False)
            a.axes.yaxis.set_visible(False)

        axes[0].imshow(img)
        axes[0].title.set_text("Image")
        axes[1].imshow(gt, cmap="gray")
        axes[1].title.set_text("Ground truth")
        axes[2].imshow(img)
        axes[2].imshow(heat, cmap="jet", alpha=0.45)
        axes[2].title.set_text("Anomaly heatmap")
        axes[3].imshow(pred, cmap="gray")
        axes[3].title.set_text("Predicted mask")
        axes[4].imshow(masked)
        axes[4].title.set_text("Predicted anomalous region")

        path = out_dir / f"{category}_{n:03d}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        written.append(path)
    return written


def render_heatmap_overlay(image_hwc_uint8: np.ndarray, score_map: np.ndarray, alpha: float = 0.45):
    """Return an RGB uint8 overlay -- used by the API and the Streamlit tool.

    ``matplotlib.cm.get_cmap`` was removed in matplotlib 3.9; ``colormaps`` is
    the supported lookup and works back to 3.5.
    """
    smin, smax = float(score_map.min()), float(score_map.max())
    norm = (score_map - smin) / (smax - smin + 1e-12)
    heat = (matplotlib.colormaps["jet"](norm)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * image_hwc_uint8.astype(np.float32) + alpha * heat.astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)

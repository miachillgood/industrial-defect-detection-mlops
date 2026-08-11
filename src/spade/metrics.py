"""Evaluation metrics for MVTec AD.

The two headline numbers checked here are the ones the public baseline
implementation reports:

* **image-level ROC-AUC** -- one score per test image vs. the good/anomalous label
* **pixel-level ROC-AUC** -- every pixel of every test image pooled together,
  scored against the ground-truth defect masks

``PRO`` (per-region overlap) is included as an extra because it is the metric the
anomaly-segmentation literature moved to; it is *not* part of the 96.4 % figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


@dataclass
class CategoryMetrics:
    category: str
    image_rocauc: float
    pixel_rocauc: float
    n_train: int
    n_test: int
    n_anomalous: int
    optimal_threshold: float
    max_f1: float
    pixel_pro: float | None = None
    image_roc_curve: dict = field(default_factory=dict, repr=False)
    pixel_roc_curve: dict = field(default_factory=dict, repr=False)

    def to_row(self) -> dict:
        return {
            "category": self.category,
            "image_rocauc": round(self.image_rocauc * 100, 2),
            "pixel_rocauc": round(self.pixel_rocauc * 100, 2),
            "pixel_pro": None if self.pixel_pro is None else round(self.pixel_pro * 100, 2),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_anomalous": self.n_anomalous,
            "optimal_threshold": round(float(self.optimal_threshold), 6),
            "max_f1": round(float(self.max_f1), 4),
        }


def _subsample_curve(fpr: np.ndarray, tpr: np.ndarray, max_points: int = 512) -> dict:
    """Keep ROC curves small enough to store in JSON without losing their shape."""
    if len(fpr) <= max_points:
        idx = np.arange(len(fpr))
    else:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}


def image_level_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores).ravel()
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(roc_auc_score(labels, scores)), _subsample_curve(fpr, tpr)


def pixel_level_metrics(
    masks: np.ndarray, score_maps: np.ndarray
) -> tuple[float, dict, float, float]:
    """Pooled per-pixel ROC-AUC plus the F1-optimal threshold used for overlays."""
    flat_gt = np.asarray(masks).ravel().astype(np.uint8)
    flat_scores = np.asarray(score_maps).ravel().astype(np.float64)

    fpr, tpr, _ = roc_curve(flat_gt, flat_scores)
    rocauc = float(auc(fpr, tpr))

    precision, recall, thresholds = precision_recall_curve(flat_gt, flat_scores)
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    best = int(np.argmax(f1))
    # precision_recall_curve returns one more precision/recall than thresholds
    threshold = float(thresholds[min(best, len(thresholds) - 1)])
    return rocauc, _subsample_curve(fpr, tpr), threshold, float(f1[best])


def per_region_overlap(
    masks: np.ndarray, score_maps: np.ndarray, max_fpr: float = 0.3, n_steps: int = 100
) -> float:
    """PRO score: mean per-connected-component recall, integrated up to ``max_fpr``.

    Each ground-truth defect region contributes equally regardless of its area,
    unlike pixel ROC-AUC where large defects dominate.
    """
    from skimage.measure import label as cc_label

    masks = np.asarray(masks).astype(bool)
    score_maps = np.asarray(score_maps).astype(np.float32)

    regions = []  # (image_index, boolean region mask)
    for i, m in enumerate(masks):
        m2 = m.squeeze()
        if not m2.any():
            continue
        components = cc_label(m2)
        for c in range(1, components.max() + 1):
            regions.append((i, components == c))
    if not regions:
        return float("nan")

    inverse_gt = ~masks.squeeze(1) if masks.ndim == 4 else ~masks
    n_negative = int(inverse_gt.sum())

    lo, hi = float(score_maps.min()), float(score_maps.max())
    thresholds = np.linspace(hi, lo, n_steps)

    fprs, pros = [], []
    for th in thresholds:
        pred = score_maps > th
        fpr = float((pred & inverse_gt).sum()) / max(n_negative, 1)
        overlaps = [float((pred[i] & region).sum()) / float(region.sum()) for i, region in regions]
        fprs.append(fpr)
        pros.append(float(np.mean(overlaps)))
        if fpr > max_fpr:
            break

    fprs_arr = np.asarray(fprs)
    pros_arr = np.asarray(pros)
    keep = fprs_arr <= max_fpr
    if keep.sum() < 2:
        return float("nan")
    return float(auc(fprs_arr[keep], pros_arr[keep]) / max_fpr)


def summarize(rows: list[dict]) -> dict:
    """Averages across categories, matching how the reference reports its table."""
    img = [r["image_rocauc"] for r in rows]
    pix = [r["pixel_rocauc"] for r in rows]
    pro = [r["pixel_pro"] for r in rows if r.get("pixel_pro") is not None]
    out = {
        "n_categories": len(rows),
        "mean_image_rocauc": round(float(np.mean(img)), 2) if img else None,
        "mean_pixel_rocauc": round(float(np.mean(pix)), 2) if pix else None,
    }
    if pro:
        out["mean_pixel_pro"] = round(float(np.mean(pro)), 2)
    return out


# Reference numbers from byungjae89/SPADE-pytorch's README (K=5) and, where the
# SPADE paper reports them, the paper's own pixel-level figures (K=50).
REFERENCE_IMAGE_ROCAUC = {
    "bottle": 97.2, "cable": 84.8, "capsule": 89.7, "carpet": 92.8, "grid": 47.3,
    "hazelnut": 88.1, "leather": 95.4, "metal_nut": 71.0, "pill": 80.1, "screw": 66.7,
    "tile": 96.5, "toothbrush": 88.9, "transistor": 90.3, "wood": 95.8, "zipper": 96.6,
}
REFERENCE_PIXEL_ROCAUC = {
    "bottle": 97.0, "cable": 92.3, "capsule": 98.4, "carpet": 98.9, "grid": 98.3,
    "hazelnut": 98.5, "leather": 99.3, "metal_nut": 97.1, "pill": 95.0, "screw": 99.1,
    "tile": 92.8, "toothbrush": 98.8, "transistor": 86.6, "wood": 95.3, "zipper": 98.6,
}
PAPER_PIXEL_ROCAUC = {
    "bottle": 98.4, "cable": 97.2, "capsule": 99.0, "carpet": 97.5, "grid": 93.7,
    "hazelnut": 99.1, "leather": 97.6, "metal_nut": 98.1, "pill": 96.5, "screw": 98.9,
    "tile": 87.4, "toothbrush": 97.9, "transistor": 94.1, "wood": 88.5, "zipper": 96.5,
}
REFERENCE_MEAN_IMAGE_ROCAUC = 85.4
REFERENCE_MEAN_PIXEL_ROCAUC = 96.4
PAPER_MEAN_IMAGE_ROCAUC = 85.5
PAPER_MEAN_PIXEL_ROCAUC = 96.5

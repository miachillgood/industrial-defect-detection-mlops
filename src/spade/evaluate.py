"""End-to-end SPADE evaluation over MVTec AD categories.

Usage::

    python -m spade.evaluate --categories all
    python -m spade.evaluate --categories bottle,cable --top-k 5 --device mps
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import FeatureCache, cache_key
from .config import SpadeConfig
from .data import CLASS_NAMES, MVTecDataset, resolve_root
from .features import PyramidFeatureExtractor, extract_features
from .metrics import (
    REFERENCE_IMAGE_ROCAUC,
    REFERENCE_MEAN_IMAGE_ROCAUC,
    REFERENCE_MEAN_PIXEL_ROCAUC,
    REFERENCE_PIXEL_ROCAUC,
    CategoryMetrics,
    image_level_metrics,
    per_region_overlap,
    pixel_level_metrics,
    summarize,
)
from .model import SPADE
from .visualize import plot_roc_curves, save_localization_panels


def _loader(dataset, cfg: SpadeConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )


def evaluate_category(
    category: str,
    cfg: SpadeConfig,
    extractor: PyramidFeatureExtractor,
    out_dir: Path,
    compute_pro: bool = True,
) -> tuple[CategoryMetrics, dict]:
    device = cfg.resolved_device()
    bank_dtype = getattr(torch, cfg.bank_dtype)

    train_ds = MVTecDataset(cfg.data_root, category, is_train=True, resize=cfg.resize, cropsize=cfg.cropsize)
    test_ds = MVTecDataset(cfg.data_root, category, is_train=False, resize=cfg.resize, cropsize=cfg.cropsize)

    t0 = time.perf_counter()

    # Only the train split is cached -- the same choice the public baseline
    # makes. It is 3 629 of the 5 354 images, and caching the test split would
    # also mean storing the input tensors the localisation figures need.
    cache = FeatureCache(cfg.cache_dir, enabled=cfg.cache_features)
    key = cache_key(
        category, "train", cfg.backbone, cfg.layers, cfg.resize, cfg.cropsize, cfg.bank_dtype
    )
    cached = cache.load(key)
    if cached is not None:
        train_features, _, _ = cached
        print(f"  [{category}] train features from cache ({cache.path_for(key).name})")
    else:
        train_features, train_labels, train_masks, _ = extract_features(
            extractor, _loader(train_ds, cfg), dtype=bank_dtype,
            progress_desc=f"[{category}] train features",
        )
        cache.save(key, train_features, train_labels, train_masks)

    test_features, test_labels, test_masks, test_images = extract_features(
        extractor, _loader(test_ds, cfg), dtype=bank_dtype,
        progress_desc=f"[{category}] test features", collect_inputs=True,
    )
    t_features = time.perf_counter() - t0

    model = SPADE(
        top_k=cfg.top_k,
        layers=cfg.layers,
        cropsize=cfg.cropsize,
        gaussian_sigma=cfg.gaussian_sigma,
        gallery_chunk=cfg.gallery_chunk,
        device=device,
        drop_gallery_remainder=cfg.drop_gallery_remainder,
    ).fit(train_features)

    t1 = time.perf_counter()
    result = model.predict(test_features, progress_desc=f"[{category}] localization")
    t_predict = time.perf_counter() - t1

    labels_np = test_labels.numpy()
    masks_np = test_masks.numpy()

    image_rocauc, image_curve = image_level_metrics(labels_np, result.image_scores)
    pixel_rocauc, pixel_curve, threshold, max_f1 = pixel_level_metrics(masks_np, result.score_maps)
    pro = per_region_overlap(masks_np, result.score_maps) if compute_pro else None

    if cfg.save_visualizations:
        save_localization_panels(
            test_images.numpy(), masks_np, result.score_maps, labels_np,
            threshold, out_dir / "images", category, max_panels=cfg.save_visualizations,
        )

    metrics = CategoryMetrics(
        category=category,
        image_rocauc=image_rocauc,
        pixel_rocauc=pixel_rocauc,
        n_train=len(train_ds),
        n_test=len(test_ds),
        n_anomalous=int(labels_np.sum()),
        optimal_threshold=threshold,
        max_f1=max_f1,
        pixel_pro=pro,
        image_roc_curve=image_curve,
        pixel_roc_curve=pixel_curve,
    )
    timings = {
        "feature_extraction_s": round(t_features, 2),
        "localization_s": round(t_predict, 2),
        "images_per_s_localization": round(len(test_ds) / max(t_predict, 1e-9), 2),
    }
    return metrics, timings


def _results_markdown(rows: list[dict], summary: dict, cfg: SpadeConfig) -> str:
    lines = [
        "# Defect detection results",
        "",
        f"- backbone: `{cfg.backbone}` (ImageNet **IMAGENET1K_V1** weights, frozen)",
        f"- K (nearest neighbours): **{cfg.top_k}**",
        f"- input: resize {cfg.resize} (LANCZOS) -> center-crop {cfg.cropsize}",
        f"- device: `{cfg.resolved_device()}`",
        "",
        "Reference = `byungjae89/SPADE-pytorch` README (K=5), used as a public baseline. "
        "`delta` is ours minus reference, in ROC-AUC points.",
        "",
        "## Image-level ROC-AUC (%)",
        "",
        "| category | reference | ours | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        ref = REFERENCE_IMAGE_ROCAUC.get(r["category"])
        delta = "-" if ref is None else f"{r['image_rocauc'] - ref:+.2f}"
        lines.append(f"| {r['category']} | {ref if ref is not None else '-'} | {r['image_rocauc']:.2f} | {delta} |")
    mi = summary["mean_image_rocauc"]
    lines.append(
        f"| **Average** | **{REFERENCE_MEAN_IMAGE_ROCAUC}** | **{mi:.2f}** | "
        f"**{mi - REFERENCE_MEAN_IMAGE_ROCAUC:+.2f}** |"
    )

    lines += [
        "",
        "## Pixel-level ROC-AUC (%)",
        "",
        "| category | reference | ours | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        ref = REFERENCE_PIXEL_ROCAUC.get(r["category"])
        delta = "-" if ref is None else f"{r['pixel_rocauc'] - ref:+.2f}"
        lines.append(f"| {r['category']} | {ref if ref is not None else '-'} | {r['pixel_rocauc']:.2f} | {delta} |")
    mp = summary["mean_pixel_rocauc"]
    lines.append(
        f"| **Average** | **{REFERENCE_MEAN_PIXEL_ROCAUC}** | **{mp:.2f}** | "
        f"**{mp - REFERENCE_MEAN_PIXEL_ROCAUC:+.2f}** |"
    )
    if summary.get("mean_pixel_pro") is not None:
        lines += [
            "",
            "## PRO (%)",
            "",
            "Per-region overlap, integrated to FPR <= 0.3. Every ground-truth defect "
            "region counts equally, so a missed small defect costs as much as a missed "
            "large one -- unlike pixel ROC-AUC, which large defects dominate. The public "
            "baseline does not report PRO, so there is no reference column.",
            "",
            "| category | PRO |",
            "| --- | ---: |",
        ]
        for r in rows:
            value = r.get("pixel_pro")
            lines.append(f"| {r['category']} | {'-' if value is None else f'{value:.2f}'} |")
        lines.append(f"| **Average** | **{summary['mean_pixel_pro']:.2f}** |")
    lines.append("")
    return "\n".join(lines)


def run(cfg: SpadeConfig, run_name: str | None = None, use_mlflow: bool = True,
        compute_pro: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    categories = list(cfg.categories) if cfg.categories else list(CLASS_NAMES)
    unknown = [c for c in categories if c not in CLASS_NAMES]
    if unknown:
        raise ValueError(f"unknown categories: {unknown}")

    root = resolve_root(cfg.data_root)
    device = cfg.resolved_device()
    run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    print(f"[spade] dataset : {root}")
    print(f"[spade] device  : {device}")
    print(f"[spade] K       : {cfg.top_k}")
    print(f"[spade] output  : {out_dir}")

    tracker = None
    if use_mlflow:
        try:
            from mlops.tracking import MlflowTracker

            tracker = MlflowTracker(experiment="spade-mvtec", run_name=run_name)
            tracker.log_params(
                {
                    **{k: v for k, v in cfg.to_dict().items() if k != "categories"},
                    "n_categories": len(categories),
                    "torch": torch.__version__,
                    "platform": platform.platform(),
                }
            )
        except Exception as exc:  # pragma: no cover - tracking must never break a run
            print(f"[spade] MLflow disabled ({exc})")
            tracker = None

    extractor = PyramidFeatureExtractor(cfg.backbone, cfg.layers, device=device)

    rows: list[dict] = []
    curves: dict[str, dict] = {}
    timings: dict[str, dict] = {}
    started = time.perf_counter()

    try:
        for category in categories:
            metrics, timing = evaluate_category(category, cfg, extractor, out_dir, compute_pro)
            row = metrics.to_row()
            rows.append(row)
            curves[category] = {"image": metrics.image_roc_curve, "pixel": metrics.pixel_roc_curve}
            timings[category] = timing
            print(
                f"  {category:<12} image ROCAUC {row['image_rocauc']:6.2f}  "
                f"pixel ROCAUC {row['pixel_rocauc']:6.2f}  "
                f"({timing['feature_extraction_s']:.0f}s feat + {timing['localization_s']:.0f}s loc)"
            )
            if tracker:
                tracker.log_metrics(
                    {
                        f"image_rocauc/{category}": row["image_rocauc"],
                        f"pixel_rocauc/{category}": row["pixel_rocauc"],
                        f"pixel_pro/{category}": row.get("pixel_pro"),
                    }
                )
    finally:
        extractor.close()

    summary = summarize(rows)
    summary["wall_clock_s"] = round(time.perf_counter() - started, 1)
    summary["reference_mean_image_rocauc"] = REFERENCE_MEAN_IMAGE_ROCAUC
    summary["reference_mean_pixel_rocauc"] = REFERENCE_MEAN_PIXEL_ROCAUC
    if summary["mean_image_rocauc"] is not None:
        summary["delta_image_rocauc"] = round(summary["mean_image_rocauc"] - REFERENCE_MEAN_IMAGE_ROCAUC, 2)
        summary["delta_pixel_rocauc"] = round(summary["mean_pixel_rocauc"] - REFERENCE_MEAN_PIXEL_ROCAUC, 2)

    payload = {
        "run_name": run_name,
        "config": cfg.to_dict(),
        "summary": summary,
        "per_category": rows,
        "timings": timings,
        "environment": {"torch": torch.__version__, "platform": platform.platform()},
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # A flat file for `dvc metrics show` / `dvc metrics diff`.
    (out_dir / "metrics.json").write_text(
        json.dumps({k: v for k, v in summary.items() if isinstance(v, (int, float))}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "roc_curves.json").write_text(json.dumps(curves), encoding="utf-8")
    md = _results_markdown(rows, summary, cfg)
    (out_dir / "results.md").write_text(md, encoding="utf-8")
    plot_roc_curves(rows, curves, out_dir / "roc_curve.png")

    print("\n" + "=" * 64)
    print(f"Average image ROCAUC : {summary['mean_image_rocauc']:.2f} %  "
          f"(reference {REFERENCE_MEAN_IMAGE_ROCAUC})")
    print(f"Average pixel ROCAUC : {summary['mean_pixel_rocauc']:.2f} %  "
          f"(reference {REFERENCE_MEAN_PIXEL_ROCAUC})")
    if summary.get("mean_pixel_pro") is not None:
        print(f"Average pixel PRO    : {summary['mean_pixel_pro']:.2f} %  (no public reference)")
    print("=" * 64)

    if tracker:
        tracker.log_metrics(
            {
                "mean_image_rocauc": summary["mean_image_rocauc"],
                "mean_pixel_rocauc": summary["mean_pixel_rocauc"],
                "mean_pixel_pro": summary.get("mean_pixel_pro"),
                "wall_clock_s": summary["wall_clock_s"],
            }
        )
        tracker.log_artifacts(out_dir)
        tracker.end()

    return payload


def _bool_arg(value: str) -> bool:
    """Parse a boolean flag that may also be given a value.

    ``--compute-pro`` alone means true; ``--compute-pro false`` is how the DVC
    pipeline passes ``${evaluate.compute_pro}`` through, since dvc.yaml has no
    conditionals and must always emit the flag.
    """
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser("spade.evaluate", description="Score MVTec AD categories with the SPADE detector")
    p.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    p.add_argument("--categories", default="all", help="'all' or a comma-separated list")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gallery-chunk", type=int, default=4096)
    p.add_argument("--bank-dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--output-dir", default="artifacts/runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save-visualizations", type=int, default=5)
    p.add_argument(
        "--cache-features",
        nargs="?",
        const=True,
        default=False,
        type=_bool_arg,
        metavar="{true,false}",
        help="reuse extracted train features across runs (~10 GB for all 15 categories)",
    )
    p.add_argument("--cache-dir", default="artifacts/cache/features")
    p.add_argument(
        "--drop-gallery-remainder",
        nargs="?",
        const=True,
        default=False,
        type=_bool_arg,
        metavar="{true,false}",
        help="reproduce the public baseline's integer-division gallery truncation "
             "(it drops the last gallery_size %% 100 rows)",
    )
    p.add_argument(
        "--compute-pro",
        nargs="?",
        const=True,
        default=True,
        type=_bool_arg,
        metavar="{true,false}",
        help="compute the PRO metric (default on; measured at ~18 s across all 15 "
             "categories, ~2 %% of a run -- pass false to skip it)",
    )
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    categories = () if args.categories == "all" else tuple(
        c.strip() for c in args.categories.split(",") if c.strip()
    )
    cfg = SpadeConfig(
        data_root=args.data_root,
        top_k=args.top_k,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gallery_chunk=args.gallery_chunk,
        bank_dtype=args.bank_dtype,
        output_dir=args.output_dir,
        save_visualizations=args.save_visualizations,
        drop_gallery_remainder=args.drop_gallery_remainder,
        cache_features=args.cache_features,
        cache_dir=args.cache_dir,
        categories=categories,
    )
    run(cfg, run_name=args.run_name, use_mlflow=not args.no_mlflow, compute_pro=args.compute_pro)


if __name__ == "__main__":
    main()

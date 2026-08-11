#!/usr/bin/env python
"""Build a deployable SPADE memory bank for one category.

SPADE has no weights to train -- the "model" is the frozen backbone plus the
feature bank of the defect-free training images. This script extracts that bank,
calibrates decision thresholds on the training split alone (leave-one-out, so no
test leakage), and writes a single ``.pt`` artifact for the API and the review
tool to load.

    python scripts/build_bank.py --category bottle
    python scripts/build_bank.py --category bottle --bank-dtype float16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spade.config import SpadeConfig, pick_device  # noqa: E402
from spade.data import MVTecDataset  # noqa: E402
from spade.features import PyramidFeatureExtractor, extract_features  # noqa: E402
from spade.model import SPADE  # noqa: E402


def build(args: argparse.Namespace) -> Path:
    device = pick_device(args.device)
    cfg = SpadeConfig(
        data_root=args.data_root,
        top_k=args.top_k,
        device=device,
        bank_dtype=args.bank_dtype,
    )

    train_ds = MVTecDataset(cfg.data_root, args.category, is_train=True,
                            resize=cfg.resize, cropsize=cfg.cropsize)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    print(f"[bank] category={args.category} n_train={len(train_ds)} device={device}")
    extractor = PyramidFeatureExtractor(cfg.backbone, cfg.layers, device=device)
    try:
        t0 = time.perf_counter()
        features, _, _, _ = extract_features(
            extractor, loader, dtype=torch.float32, progress_desc=f"[{args.category}] features"
        )
        print(f"[bank] features extracted in {time.perf_counter() - t0:.1f}s")

        model = SPADE(
            top_k=cfg.top_k, layers=cfg.layers, cropsize=cfg.cropsize,
            gaussian_sigma=cfg.gaussian_sigma, gallery_chunk=cfg.gallery_chunk, device=device,
        ).fit(features)

        # --- calibrate on the training split only, leave-one-out -------------
        print("[bank] calibrating thresholds (leave-one-out over the train split)")
        loo = model.predict(features, progress_desc=f"[{args.category}] calibration",
                            exclude_self=True)
        img_scores = loo.image_scores
        pix = loo.score_maps.reshape(-1)

        image_threshold = float(np.percentile(img_scores, args.image_percentile))
        pixel_threshold = float(np.percentile(pix, args.pixel_percentile))

        metadata = {
            "category": args.category,
            "backbone": cfg.backbone,
            "resize": cfg.resize,
            "cropsize": cfg.cropsize,
            "top_k": cfg.top_k,
            "n_train_images": len(train_ds),
            "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold,
            "calibration": {
                "method": "leave-one-out over the defect-free training split",
                "image_percentile": args.image_percentile,
                "pixel_percentile": args.pixel_percentile,
                "train_image_score_mean": float(img_scores.mean()),
                "train_image_score_std": float(img_scores.std()),
                "train_image_score_max": float(img_scores.max()),
                "train_pixel_score_max": float(pix.max()),
            },
            "torch": torch.__version__,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bank_path = out_dir / f"spade_{args.category}.pt"
        model.save_bank(bank_path, dtype=args.bank_dtype, metadata=metadata)

        (out_dir / f"spade_{args.category}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        size_mb = bank_path.stat().st_size / 2**20
        print(f"[bank] wrote {bank_path} ({size_mb:.1f} MB, dtype={args.bank_dtype})")
        print(f"[bank] image_threshold={image_threshold:.4f} pixel_threshold={pixel_threshold:.4f}")
        return bank_path
    finally:
        extractor.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="bottle")
    p.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    p.add_argument("--output-dir", default="artifacts/banks")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bank-dtype", default="float16", choices=["float32", "float16"])
    p.add_argument("--image-percentile", type=float, default=99.0,
                   help="percentile of leave-one-out train scores used as the image threshold")
    p.add_argument("--pixel-percentile", type=float, default=99.5,
                   help="percentile of leave-one-out train pixel scores used as the pixel threshold")
    build(p.parse_args())


if __name__ == "__main__":
    main()

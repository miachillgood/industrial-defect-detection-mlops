"""Single-image inference on top of a persisted SPADE memory bank.

Shared by the FastAPI service and the Streamlit review tool so both agree on
scores, thresholds and overlays.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import pick_device
from .data import build_image_transform, denormalize
from .features import PyramidFeatureExtractor
from .model import SPADE
from .visualize import render_heatmap_overlay


@dataclass
class Prediction:
    category: str
    image_score: float
    is_anomalous: bool
    image_threshold: float
    pixel_threshold: float
    anomalous_pixel_ratio: float
    score_map: np.ndarray
    neighbour_indices: list[int]
    latency_ms: float

    def to_payload(self, include_images: bool = True) -> dict:
        out = {
            "category": self.category,
            "image_score": round(float(self.image_score), 6),
            "is_anomalous": bool(self.is_anomalous),
            "image_threshold": round(float(self.image_threshold), 6),
            "pixel_threshold": round(float(self.pixel_threshold), 6),
            "anomalous_pixel_ratio": round(float(self.anomalous_pixel_ratio), 6),
            "neighbour_indices": [int(i) for i in self.neighbour_indices],
            "latency_ms": round(float(self.latency_ms), 2),
            "score_map_shape": list(self.score_map.shape),
        }
        if include_images:
            out["heatmap_png_base64"] = self.heatmap_png_base64()
            out["mask_png_base64"] = self.mask_png_base64()
        return out

    def heatmap_png_base64(self) -> str:
        smin, smax = float(self.score_map.min()), float(self.score_map.max())
        norm = (self.score_map - smin) / (smax - smin + 1e-12)
        img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        return _png_b64(img)

    def mask_png_base64(self) -> str:
        mask = (self.score_map > self.pixel_threshold).astype(np.uint8) * 255
        return _png_b64(Image.fromarray(mask, mode="L"))


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class SpadePredictor:
    """Loads one category's memory bank and scores images against it."""

    def __init__(self, bank_path: str | Path, device: str = "auto", gallery_chunk: int = 4096) -> None:
        self.bank_path = Path(bank_path)
        if not self.bank_path.exists():
            raise FileNotFoundError(
                f"memory bank not found at {self.bank_path}. "
                "Build one with `python scripts/build_bank.py --category bottle`, "
                "or pull it with `dvc pull`."
            )
        self.device = pick_device(device)
        self.model, self.metadata = SPADE.load_bank(
            self.bank_path, device=self.device, gallery_chunk=gallery_chunk
        )
        self.category = self.metadata.get("category", "unknown")
        self.image_threshold = float(self.metadata.get("image_threshold", float("inf")))
        self.pixel_threshold = float(self.metadata.get("pixel_threshold", float("inf")))
        self.cropsize = self.model.cropsize
        self.extractor = PyramidFeatureExtractor(
            self.metadata.get("backbone", "wide_resnet50_2"),
            tuple(self.model.layers),
            device=self.device,
        )
        self.transform = build_image_transform(
            self.metadata.get("resize", 256), self.cropsize
        )

    # ------------------------------------------------------------------ infer
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        import time

        t0 = time.perf_counter()
        x = self.transform(image.convert("RGB")).unsqueeze(0)
        features = self.extractor(x)
        features = {k: v.detach().cpu().float() for k, v in features.items()}
        result = self.model.predict(features)
        latency = (time.perf_counter() - t0) * 1000

        score_map = result.score_maps[0]
        return Prediction(
            category=self.category,
            image_score=float(result.image_scores[0]),
            is_anomalous=bool(result.image_scores[0] > self.image_threshold),
            image_threshold=self.image_threshold,
            pixel_threshold=self.pixel_threshold,
            anomalous_pixel_ratio=float((score_map > self.pixel_threshold).mean()),
            score_map=score_map,
            neighbour_indices=result.neighbour_indices[0].tolist(),
            latency_ms=latency,
        )

    def preprocessed_rgb(self, image: Image.Image) -> np.ndarray:
        """The 224x224 crop the model actually saw, as an HWC uint8 array."""
        return denormalize(self.transform(image.convert("RGB")))

    def overlay(self, image: Image.Image, prediction: Prediction, alpha: float = 0.45) -> np.ndarray:
        return render_heatmap_overlay(self.preprocessed_rgb(image), prediction.score_map, alpha)

    def info(self) -> dict:
        return {
            "category": self.category,
            "bank_path": str(self.bank_path),
            "device": self.device,
            "n_train_images": self.model.n_train,
            "top_k": self.model.top_k,
            "layers": list(self.model.layers),
            "cropsize": self.cropsize,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "metadata": self.metadata,
        }

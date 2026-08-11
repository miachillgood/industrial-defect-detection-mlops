"""FastAPI inference service for SPADE.

Loads one memory bank per MVTec category from ``SPADE_BANK_DIR`` and serves
anomaly scores plus localisation masks.

    uvicorn apps.api.main:app --reload --port 8000

Environment
-----------
``SPADE_BANK_DIR``   directory of ``spade_<category>.pt`` files (default ``artifacts/banks``)
``SPADE_DEVICE``     ``auto`` | ``cpu`` | ``cuda`` | ``mps`` (default ``auto``)
``SPADE_EAGER_LOAD`` ``1`` to load every bank at startup instead of on first use
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from spade import __version__ as spade_version  # noqa: E402
from spade.inference import SpadePredictor  # noqa: E402

BANK_DIR = Path(os.environ.get("SPADE_BANK_DIR", REPO_ROOT / "artifacts" / "banks"))
DEVICE = os.environ.get("SPADE_DEVICE", "auto")
MAX_UPLOAD_BYTES = int(os.environ.get("SPADE_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))


class PredictResponse(BaseModel):
    category: str
    image_score: float = Field(..., description="mean distance to the K nearest training images")
    is_anomalous: bool
    image_threshold: float
    pixel_threshold: float
    anomalous_pixel_ratio: float = Field(..., description="fraction of pixels above the pixel threshold")
    neighbour_indices: list[int]
    latency_ms: float
    score_map_shape: list[int]
    heatmap_png_base64: str | None = None
    mask_png_base64: str | None = None


class PredictorRegistry:
    """Lazily loads and caches one predictor per category. Thread-safe."""

    def __init__(self, bank_dir: Path, device: str = "auto") -> None:
        self.bank_dir = Path(bank_dir)
        self.device = device
        self._cache: dict[str, SpadePredictor] = {}
        self._lock = threading.Lock()

    def available(self) -> list[str]:
        if not self.bank_dir.is_dir():
            return []
        return sorted(p.stem.removeprefix("spade_") for p in self.bank_dir.glob("spade_*.pt"))

    def loaded(self) -> list[str]:
        with self._lock:
            return sorted(self._cache)

    def get(self, category: str) -> SpadePredictor:
        with self._lock:
            if category not in self._cache:
                path = self.bank_dir / f"spade_{category}.pt"
                if not path.exists():
                    raise KeyError(category)
                t0 = time.perf_counter()
                self._cache[category] = SpadePredictor(path, device=self.device)
                print(f"[api] loaded bank '{category}' in {time.perf_counter() - t0:.1f}s")
            return self._cache[category]

    def load_all(self) -> None:
        for category in self.available():
            try:
                self.get(category)
            except Exception as exc:  # pragma: no cover
                print(f"[api] failed to load '{category}': {exc}")


registry = PredictorRegistry(BANK_DIR, DEVICE)
STATS = {"requests": 0, "anomalous": 0, "errors": 0, "total_latency_ms": 0.0}
_stats_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(f"[api] bank dir: {BANK_DIR}")
    print(f"[api] available categories: {registry.available() or '(none -- run scripts/build_bank.py)'}")
    if os.environ.get("SPADE_EAGER_LOAD") == "1":
        registry.load_all()
    yield


app = FastAPI(
    title="SPADE industrial defect detection API",
    description=(
        "Anomaly detection and localisation with SPADE "
        "(Cohen & Hoshen, arXiv:2005.02357), configured after "
        "byungjae89/SPADE-pytorch. Not affiliated with NVlabs/SPADE."
    ),
    version=spade_version,
    lifespan=lifespan,
)


def _read_image(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
    try:
        return Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise HTTPException(400, "uploaded file is not a readable image") from exc


def get_predictor(category: str = Query("bottle", description="MVTec category")) -> SpadePredictor:
    try:
        return registry.get(category)
    except KeyError as exc:
        raise HTTPException(
            404,
            f"no memory bank for category {category!r}. "
            f"Available: {registry.available()}. Build one with scripts/build_bank.py.",
        ) from exc


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": spade_version,
        "bank_dir": str(BANK_DIR),
        "available_categories": registry.available(),
        "loaded_categories": registry.loaded(),
    }


@app.get("/models")
def models() -> dict:
    out = {}
    for category in registry.available():
        meta_path = BANK_DIR / f"spade_{category}.json"
        if meta_path.exists():
            import json

            out[category] = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            out[category] = {"category": category, "metadata": "not available until loaded"}
    return {"models": out}


@app.get("/stats")
def stats() -> dict:
    with _stats_lock:
        s = dict(STATS)
    s["mean_latency_ms"] = round(s["total_latency_ms"] / s["requests"], 2) if s["requests"] else 0.0
    return s


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="image of the part under inspection"),
    include_images: bool = Query(True, description="embed heatmap/mask PNGs as base64"),
    predictor: SpadePredictor = Depends(get_predictor),
) -> JSONResponse:
    image = _read_image(await file.read())
    try:
        result = predictor.predict(image)
    except Exception as exc:  # pragma: no cover
        with _stats_lock:
            STATS["errors"] += 1
        raise HTTPException(500, f"inference failed: {exc}") from exc

    with _stats_lock:
        STATS["requests"] += 1
        STATS["anomalous"] += int(result.is_anomalous)
        STATS["total_latency_ms"] += result.latency_ms
    return JSONResponse(result.to_payload(include_images=include_images))


@app.post("/predict/overlay", response_class=Response)
async def predict_overlay(
    file: UploadFile = File(...),
    alpha: float = Query(0.45, ge=0.0, le=1.0),
    predictor: SpadePredictor = Depends(get_predictor),
) -> Response:
    """Return the heatmap blended over the pre-processed crop, as a PNG."""
    image = _read_image(await file.read())
    result = predictor.predict(image)
    overlay = predictor.overlay(image, result, alpha=alpha)

    buf = io.BytesIO()
    Image.fromarray(np.asarray(overlay)).save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Image-Score": f"{result.image_score:.6f}",
            "X-Is-Anomalous": str(result.is_anomalous).lower(),
            "X-Latency-Ms": f"{result.latency_ms:.2f}",
        },
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "spade-defect-detection",
        "docs": "/docs",
        "endpoints": ["/health", "/models", "/stats", "/predict", "/predict/overlay"],
    }

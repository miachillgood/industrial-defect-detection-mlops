"""End-to-end API tests against a synthetic memory bank.

These exercise the real backbone (ImageNet weights are downloaded/cached by
torchvision), so they are marked ``needs_backbone``. Run only the fast tests
with ``pytest -m "not needs_backbone"``.
"""

from __future__ import annotations

import base64
import importlib
import io

import numpy as np
import pytest
import torch
from PIL import Image

from spade.model import SPADE

pytestmark = pytest.mark.needs_backbone


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    bank_dir = tmp_path_factory.mktemp("banks")
    torch.manual_seed(0)
    n = 4
    bank = {
        "layer1": torch.rand(n, 256, 56, 56),
        "layer2": torch.rand(n, 512, 28, 28),
        "layer3": torch.rand(n, 1024, 14, 14),
        "avgpool": torch.rand(n, 2048, 1, 1),
    }
    model = SPADE(top_k=2, cropsize=224).fit(bank)
    model.save_bank(
        bank_dir / "spade_bottle.pt",
        dtype="float16",
        metadata={
            "category": "bottle",
            "backbone": "wide_resnet50_2",
            "resize": 256,
            "cropsize": 224,
            "image_threshold": 1.0,
            "pixel_threshold": 1.0,
            "n_train_images": n,
        },
    )

    import os

    os.environ["SPADE_BANK_DIR"] = str(bank_dir)
    os.environ["SPADE_DEVICE"] = "cpu"
    os.environ.pop("SPADE_EAGER_LOAD", None)

    import apps.api.main as api_main

    importlib.reload(api_main)
    with fastapi_testclient.TestClient(api_main.app) as client:
        yield client


def _png_bytes(color: int = 128, size: int = 256) -> bytes:
    arr = np.full((size, size, 3), color, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health_lists_the_bank(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "bottle" in body["available_categories"]


def test_models_endpoint(api_client):
    r = api_client.get("/models")
    assert r.status_code == 200
    assert "bottle" in r.json()["models"]


def test_predict_returns_scores_and_images(api_client):
    r = api_client.post(
        "/predict?category=bottle",
        files={"file": ("part.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["category"] == "bottle"
    assert body["image_score"] > 0
    assert isinstance(body["is_anomalous"], bool)
    assert body["score_map_shape"] == [224, 224]
    assert len(body["neighbour_indices"]) == 2

    heat = Image.open(io.BytesIO(base64.b64decode(body["heatmap_png_base64"])))
    assert heat.size == (224, 224)
    mask = Image.open(io.BytesIO(base64.b64decode(body["mask_png_base64"])))
    assert mask.size == (224, 224)


def test_predict_can_skip_images(api_client):
    r = api_client.post(
        "/predict?category=bottle&include_images=false",
        files={"file": ("part.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    assert r.json().get("heatmap_png_base64") is None


def test_predict_overlay_returns_png(api_client):
    r = api_client.post(
        "/predict/overlay?category=bottle",
        files={"file": ("part.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert "X-Image-Score" in r.headers
    assert Image.open(io.BytesIO(r.content)).size == (224, 224)


def test_unknown_category_is_404(api_client):
    r = api_client.post(
        "/predict?category=hazelnut",
        files={"file": ("part.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 404
    assert "hazelnut" in r.text


def test_non_image_upload_is_400(api_client):
    r = api_client.post(
        "/predict?category=bottle",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert r.status_code == 400


def test_stats_counts_requests(api_client):
    before = api_client.get("/stats").json()["requests"]
    api_client.post(
        "/predict?category=bottle&include_images=false",
        files={"file": ("part.png", _png_bytes(), "image/png")},
    )
    after = api_client.get("/stats").json()
    assert after["requests"] == before + 1
    assert after["mean_latency_ms"] > 0

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = REPO_ROOT / "data" / "mvtec_anomaly_detection"


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: requires the MVTec AD dataset on disk")
    config.addinivalue_line("markers", "needs_backbone: downloads/loads ImageNet weights")


@pytest.fixture(scope="session")
def data_root() -> Path:
    if not (DATA_ROOT / "bottle" / "train" / "good").is_dir():
        pytest.skip("MVTec AD not present; run scripts/prepare_data.py")
    return DATA_ROOT


@pytest.fixture
def synthetic_bank():
    """A tiny, correctly-shaped feature bank: 6 near-identical 'good' samples."""
    torch.manual_seed(0)
    n, cropsize = 6, 224
    base = {
        "layer1": torch.rand(1, 8, 56, 56),
        "layer2": torch.rand(1, 16, 28, 28),
        "layer3": torch.rand(1, 32, 14, 14),
        "avgpool": torch.rand(1, 64, 1, 1),
    }
    bank = {k: (v.repeat(n, 1, 1, 1) + 0.01 * torch.randn(n, *v.shape[1:])) for k, v in base.items()}
    return bank, base, cropsize

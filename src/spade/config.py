"""Configuration for the defect-detection pipeline.

Defaults mirror ``byungjae89/SPADE-pytorch`` so that a default run lands on the
85.4 % image-level / 96.4 % pixel-level ROC-AUC reported by that repository.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


def pick_device(preference: str = "auto") -> str:
    """Resolve ``auto`` to the best locally available backend."""
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SpadeConfig:
    # --- data ---------------------------------------------------------------
    data_root: str = "data/mvtec_anomaly_detection"
    resize: int = 256
    cropsize: int = 224

    # --- backbone -----------------------------------------------------------
    # ``wide_resnet50_2`` with the original torchvision ImageNet weights
    # (IMAGENET1K_V1 -- what ``pretrained=True`` used to resolve to).
    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer1", "layer2", "layer3")

    # --- SPADE hyper-parameters --------------------------------------------
    # NOTE: the paper uses K=50; the reference PyTorch implementation we
    # public baseline uses K=5, and so do we by default.
    top_k: int = 5
    gaussian_sigma: float = 4.0

    # --- runtime ------------------------------------------------------------
    device: str = "auto"
    batch_size: int = 32
    num_workers: int = 4
    # Gallery rows compared against the test feature map at once. Purely a
    # memory/speed knob -- it does not change the result.
    gallery_chunk: int = 4096
    bank_dtype: str = "float32"
    seed: int = 42

    # --- outputs ------------------------------------------------------------
    output_dir: str = "artifacts/runs"
    save_visualizations: int = 5

    categories: tuple[str, ...] = field(default_factory=tuple)

    def resolved_device(self) -> str:
        return pick_device(self.device)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = list(self.layers)
        d["categories"] = list(self.categories)
        d["resolved_device"] = self.resolved_device()
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SpadeConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw.pop("resolved_device", None)
        raw["layers"] = tuple(raw.get("layers", ("layer1", "layer2", "layer3")))
        raw["categories"] = tuple(raw.get("categories", ()))
        return cls(**raw)

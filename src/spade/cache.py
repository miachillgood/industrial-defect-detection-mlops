"""On-disk cache for extracted backbone features.

The public baseline implementation pickles each category's *training* features so
a second run skips re-extraction. That matters more than it looks: features
depend only on (dataset, backbone, pre-processing) -- never on ``K`` or the
Gaussian sigma -- so a sweep over ``K`` re-extracts identical tensors every time.
Extraction is ~80 % of a run's wall-clock.

Differences from the baseline's version:

* the cache key is derived from everything that can change the tensors
  (category, split, backbone, layer set, resize/crop, dtype, torch version),
  so a stale cache cannot silently poison a run -- the baseline keys on the
  class name alone, which goes wrong the moment you change the input size;
* ``torch.save`` instead of ``pickle``, and off by default, because the full
  15-category train cache is ~10 GB at float16.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

DEFAULT_CACHE_DIR = Path("artifacts/cache/features")


def cache_key(
    category: str,
    split: str,
    backbone: str,
    layers: tuple[str, ...],
    resize: int,
    cropsize: int,
    dtype: str,
) -> str:
    payload = {
        "category": category,
        "split": split,
        "backbone": backbone,
        "layers": list(layers),
        "resize": resize,
        "cropsize": cropsize,
        "dtype": dtype,
        # Feature values are not guaranteed stable across torch releases.
        "torch": torch.__version__,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return f"{category}_{split}_{digest}"


class FeatureCache:
    """Content-addressed store for one (category, split) feature bundle."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR, enabled: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

    def path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    def load(self, key: str):
        """Return ``(features, labels, masks)`` or ``None`` on a miss."""
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # a truncated or unreadable cache is just a miss
            print(f"[cache] ignoring unreadable {path.name}: {exc}")
            return None
        return payload["features"], payload["labels"], payload["masks"]

    def save(self, key: str, features: dict, labels, masks) -> Path | None:
        if not self.enabled:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"features": features, "labels": labels, "masks": masks}, tmp)
        # Atomic replace so an interrupted write never leaves a half-file that a
        # later run would have to guess about.
        tmp.replace(path)
        return path

    def size_bytes(self) -> int:
        if not self.cache_dir.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.cache_dir.glob("*.pt"))

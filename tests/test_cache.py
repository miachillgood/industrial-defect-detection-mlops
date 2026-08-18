"""Feature cache.

The cache exists so that sweeping ``K`` does not re-extract identical tensors.
That only holds if the key covers everything that *can* change the tensors --
a cache that silently returns stale features is worse than no cache at all.
"""

from __future__ import annotations

import torch

from spade.cache import FeatureCache, cache_key

BASE = dict(
    category="bottle",
    split="train",
    backbone="wide_resnet50_2",
    layers=("layer1", "layer2", "layer3"),
    resize=256,
    cropsize=224,
    dtype="float32",
)


def test_key_is_stable_for_identical_inputs():
    assert cache_key(**BASE) == cache_key(**BASE)


def test_key_starts_with_a_human_readable_prefix():
    assert cache_key(**BASE).startswith("bottle_train_")


def test_every_input_dimension_changes_the_key():
    """Guards the whole point of the cache: no silent stale hits."""
    baseline = cache_key(**BASE)
    variants = {
        "category": {**BASE, "category": "cable"},
        "split": {**BASE, "split": "test"},
        "backbone": {**BASE, "backbone": "resnet18"},
        "layers": {**BASE, "layers": ("layer1", "layer2")},
        "resize": {**BASE, "resize": 512},
        "cropsize": {**BASE, "cropsize": 320},
        "dtype": {**BASE, "dtype": "float16"},
    }
    for name, kwargs in variants.items():
        assert cache_key(**kwargs) != baseline, f"changing {name} must change the cache key"


def test_disabled_cache_is_a_no_op(tmp_path):
    cache = FeatureCache(tmp_path, enabled=False)
    assert cache.save("k", {"layer1": torch.rand(2, 3)}, torch.tensor([0, 1]), torch.zeros(2)) is None
    assert cache.load("k") is None
    assert not list(tmp_path.glob("*.pt"))


def test_roundtrip_preserves_tensors(tmp_path):
    cache = FeatureCache(tmp_path, enabled=True)
    features = {"layer1": torch.rand(4, 8, 5, 5), "avgpool": torch.rand(4, 16, 1, 1)}
    labels = torch.tensor([0, 0, 1, 1])
    masks = torch.zeros(4, 1, 8, 8, dtype=torch.uint8)

    cache.save("demo", features, labels, masks)
    loaded = cache.load("demo")
    assert loaded is not None

    got_features, got_labels, got_masks = loaded
    assert sorted(got_features) == sorted(features)
    for k in features:
        assert torch.equal(got_features[k], features[k])
    assert torch.equal(got_labels, labels)
    assert torch.equal(got_masks, masks)


def test_miss_on_unknown_key(tmp_path):
    assert FeatureCache(tmp_path, enabled=True).load("never-written") is None


def test_corrupt_file_is_a_miss_not_a_crash(tmp_path):
    """A killed run must not poison every later run."""
    cache = FeatureCache(tmp_path, enabled=True)
    cache.path_for("torn").write_bytes(b"not a torch archive")
    assert cache.load("torn") is None


def test_no_temp_file_survives_a_successful_save(tmp_path):
    cache = FeatureCache(tmp_path, enabled=True)
    cache.save("demo", {"layer1": torch.rand(2, 2)}, torch.tensor([0, 1]), torch.zeros(2))
    assert not list(tmp_path.glob("*.tmp"))
    assert cache.path_for("demo").exists()


def test_size_bytes_reports_zero_for_a_missing_dir(tmp_path):
    assert FeatureCache(tmp_path / "nope", enabled=True).size_bytes() == 0


def test_size_bytes_counts_written_entries(tmp_path):
    cache = FeatureCache(tmp_path, enabled=True)
    cache.save("a", {"layer1": torch.rand(16, 16)}, torch.tensor([0]), torch.zeros(1))
    assert cache.size_bytes() > 0

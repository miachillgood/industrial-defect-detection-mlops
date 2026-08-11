"""Dataset-layer tests. Most need the real MVTec AD tree on disk."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from spade.data import (
    CLASS_NAMES,
    MVTecDataset,
    build_image_transform,
    build_mask_transform,
    denormalize,
    resolve_root,
)


def test_class_names_are_the_official_fifteen():
    assert len(CLASS_NAMES) == 15
    assert len(set(CLASS_NAMES)) == 15
    assert "bottle" in CLASS_NAMES and "zipper" in CLASS_NAMES


def test_transform_shapes():
    img = Image.fromarray(np.random.randint(0, 255, (900, 900, 3), dtype=np.uint8))
    x = build_image_transform(256, 224)(img)
    assert x.shape == (3, 224, 224)

    mask = Image.fromarray((np.random.rand(900, 900) > 0.5).astype(np.uint8) * 255)
    m = build_mask_transform(256, 224)(mask)
    assert m.shape == (1, 224, 224)


def test_denormalize_roundtrips_to_uint8():
    x = torch.rand(3, 32, 32)
    out = denormalize(x)
    assert out.dtype == np.uint8
    assert out.shape == (32, 32, 3)


def test_resolve_root_rejects_a_wrong_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_root(tmp_path)


@pytest.mark.needs_data
def test_train_split_is_all_good(data_root):
    ds = MVTecDataset(data_root, "bottle", is_train=True)
    assert len(ds) == 209
    assert set(ds.labels) == {0}
    assert all(m is None for m in ds.mask_paths)


@pytest.mark.needs_data
def test_test_split_has_both_classes_and_masks(data_root):
    ds = MVTecDataset(data_root, "bottle", is_train=False)
    summary = ds.summary()
    assert summary["n_images"] == len(ds)
    assert summary["n_anomalous"] > 0
    assert summary["n_good"] > 0
    for label, mask_path in zip(ds.labels, ds.mask_paths, strict=True):
        assert (mask_path is None) == (label == 0)


@pytest.mark.needs_data
def test_items_have_the_right_shapes_and_binary_masks(data_root):
    ds = MVTecDataset(data_root, "bottle", is_train=False)
    anomalous_idx = next(i for i, y in enumerate(ds.labels) if y == 1)
    good_idx = next(i for i, y in enumerate(ds.labels) if y == 0)

    x, y, mask = ds[anomalous_idx]
    assert x.shape == (3, 224, 224) and y == 1 and mask.shape == (1, 224, 224)
    assert set(np.unique(mask.numpy())).issubset({0.0, 1.0})
    assert mask.sum() > 0

    x, y, mask = ds[good_idx]
    assert y == 0 and mask.sum() == 0


@pytest.mark.needs_data
def test_unknown_category_rejected(data_root):
    with pytest.raises(ValueError):
        MVTecDataset(data_root, "not_a_category")


@pytest.mark.needs_data
@pytest.mark.parametrize("category", CLASS_NAMES)
def test_every_category_loads(data_root, category):
    train = MVTecDataset(data_root, category, is_train=True)
    test = MVTecDataset(data_root, category, is_train=False)
    assert len(train) > 0 and len(test) > 0
    assert set(train.labels) == {0}
    assert sum(test.labels) > 0

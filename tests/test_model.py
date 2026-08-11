"""Core SPADE algorithm tests -- these run without the dataset or the backbone."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spade.model import EPS, SPADE, _pairwise_min_distance


def test_pairwise_min_distance_matches_bruteforce():
    torch.manual_seed(1)
    query = torch.randn(17, 9)
    gallery = torch.randn(53, 9)
    got = _pairwise_min_distance(query, gallery, chunk=7)
    expected = torch.cdist(query, gallery).min(dim=1).values
    assert torch.allclose(got, expected, atol=1e-4)


def test_chunking_does_not_change_the_result():
    torch.manual_seed(2)
    query = torch.randn(11, 5)
    gallery = torch.randn(100, 5)
    a = _pairwise_min_distance(query, gallery, chunk=3)
    b = _pairwise_min_distance(query, gallery, chunk=1000)
    assert torch.allclose(a, b, atol=1e-5)


def test_channelwise_reduction_is_the_intended_distance():
    """Regression guard for the PyTorch 2.x ``pairwise_distance`` trap.

    The reference implementation relies on ``torch.pairwise_distance`` reducing
    over ``dim=1`` (channels), which was true in PyTorch 1.x. PyTorch >= 2.0
    reduces over the last dim instead. If someone "simplifies" our explicit
    channel-axis reduction back to ``torch.pairwise_distance``, this fails.
    """
    c, h, w = 4, 3, 5
    gallery = torch.randn(6, c)
    query_map = torch.randn(c, h, w)

    query = query_map.reshape(c, h * w).t()
    got = _pairwise_min_distance(query, gallery + EPS).reshape(h, w)

    expected = torch.empty(h, w)
    for i in range(h):
        for j in range(w):
            v = query_map[:, i, j]
            expected[i, j] = min(float(torch.norm(g + EPS - v)) for g in gallery)

    assert torch.allclose(got, expected, atol=1e-4)
    # And the modern built-in genuinely disagrees, which is why we avoid it.
    modern = torch.pairwise_distance(gallery.unsqueeze(-1).unsqueeze(-1), query_map.unsqueeze(0))
    assert modern.shape != (gallery.shape[0], h, w)


def test_identical_image_scores_near_zero(synthetic_bank):
    bank, base, cropsize = synthetic_bank
    model = SPADE(top_k=3, layers=("layer1", "layer2", "layer3"), cropsize=cropsize).fit(bank)
    # Score one of the bank members back against the bank.
    query = {k: v[0:1] for k, v in bank.items()}
    out = model.predict(query)
    assert out.image_scores.shape == (1,)
    assert out.score_maps.shape == (1, cropsize, cropsize)
    assert out.image_scores[0] < 1.0
    assert 0 in out.neighbour_indices[0]


def test_injected_defect_raises_the_local_score(synthetic_bank):
    bank, base, cropsize = synthetic_bank
    model = SPADE(top_k=3, cropsize=cropsize, gaussian_sigma=1.0).fit(bank)

    clean = {k: v.clone() for k, v in base.items()}
    defective = {k: v.clone() for k, v in base.items()}
    # A blob of out-of-distribution activation in the top-left of layer1.
    defective["layer1"][0, :, 4:12, 4:12] += 8.0

    clean_out = model.predict(clean)
    defect_out = model.predict(defective)

    # layer1 position (8, 8) of 56 maps to roughly (32, 32) of 224.
    region = slice(8, 56)
    assert defect_out.score_maps[0][region, region].max() > clean_out.score_maps[0].max() * 2
    assert defect_out.score_maps[0].mean() > clean_out.score_maps[0].mean()


def test_topk_is_clamped_to_bank_size(synthetic_bank):
    bank, base, cropsize = synthetic_bank
    model = SPADE(top_k=50, cropsize=cropsize).fit(bank)
    out = model.predict(base)
    assert out.neighbour_indices.shape[1] == model.n_train


def test_exclude_self_ignores_the_diagonal(synthetic_bank):
    bank, _, cropsize = synthetic_bank
    model = SPADE(top_k=2, cropsize=cropsize).fit(bank)
    out = model.predict(bank, exclude_self=True)
    for i, neighbours in enumerate(out.neighbour_indices):
        assert i not in neighbours
    # Leave-one-out scores must be strictly positive (no zero-distance self match).
    assert (out.image_scores > 0).all()


def test_exclude_self_requires_square_matrix(synthetic_bank):
    bank, base, cropsize = synthetic_bank
    model = SPADE(cropsize=cropsize).fit(bank)
    with pytest.raises(ValueError):
        model.predict(base, exclude_self=True)


def test_fit_rejects_missing_layers(synthetic_bank):
    bank, _, _ = synthetic_bank
    partial = {k: v for k, v in bank.items() if k != "layer2"}
    with pytest.raises(KeyError):
        SPADE().fit(partial)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SPADE().predict({})


def test_bank_roundtrip(tmp_path, synthetic_bank):
    bank, base, cropsize = synthetic_bank
    model = SPADE(top_k=3, cropsize=cropsize).fit(bank)
    path = tmp_path / "bank.pt"
    model.save_bank(path, dtype="float16", metadata={"category": "synthetic"})

    restored, meta = SPADE.load_bank(path)
    assert meta["category"] == "synthetic"
    assert restored.n_train == model.n_train
    assert restored.top_k == 3

    a = model.predict(base).image_scores
    b = restored.predict(base).image_scores
    np.testing.assert_allclose(a, b, rtol=2e-2, atol=2e-2)


def test_drop_gallery_remainder_quirk_is_opt_in(synthetic_bank):
    """The reference drops ``gallery % 100`` rows; we keep them unless asked."""
    bank, base, cropsize = synthetic_bank
    faithful = SPADE(top_k=3, cropsize=cropsize, drop_gallery_remainder=True).fit(bank)
    ours = SPADE(top_k=3, cropsize=cropsize, drop_gallery_remainder=False).fit(bank)
    # A smaller gallery can only ever increase the minimum distance.
    assert faithful.predict(base).score_maps.mean() >= ours.predict(base).score_maps.mean() - 1e-6

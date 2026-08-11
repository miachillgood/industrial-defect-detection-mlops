from __future__ import annotations

import pytest

from mlops.review_store import ReviewRecord, ReviewStore


def _record(path: str, human: str, model: str, **kw) -> ReviewRecord:
    return ReviewRecord(
        image_path=path,
        category="bottle",
        human_verdict=human,
        model_score=kw.pop("score", 1.0),
        model_threshold=0.5,
        model_verdict=model,
        **kw,
    )


def test_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        _record("a.png", "broken", "ok")


def test_append_and_load_roundtrip(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    store.append(_record("a.png", "ok", "ok"))
    store.append(_record("b.png", "defect", "defect", reviewer="qa-2"))
    rows = store.load()
    assert len(rows) == 2
    assert rows[1]["reviewer"] == "qa-2"


def test_latest_decision_wins(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    store.append(_record("a.png", "ok", "defect"))
    store.append(_record("a.png", "defect", "defect", notes="looked again"))
    latest = store.latest_per_image()
    assert len(latest) == 1
    assert latest["a.png"]["human_verdict"] == "defect"
    assert latest["a.png"]["notes"] == "looked again"
    # history is preserved
    assert len(store.load()) == 2


def test_stats_confusion_and_f1(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    store.append(_record("tp.png", "defect", "defect"))
    store.append(_record("tn.png", "ok", "ok"))
    store.append(_record("fp.png", "ok", "defect"))
    store.append(_record("fn.png", "defect", "ok"))
    store.append(_record("u.png", "unsure", "ok"))

    s = store.stats()
    assert s["n_reviewed"] == 5
    assert s["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx(0.5)
    assert s["n_unsure"] == 1
    assert s["agreement_rate"] == pytest.approx(2 / 5)


def test_stats_on_empty_store(tmp_path):
    assert ReviewStore(tmp_path / "nope.jsonl").stats() == {"n_reviewed": 0}


def test_tolerates_torn_trailing_line(tmp_path):
    path = tmp_path / "reviews.jsonl"
    store = ReviewStore(path)
    store.append(_record("a.png", "ok", "ok"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"image_path": "b.png", "cat')  # interrupted write
    assert len(store.load()) == 1


def test_export_csv(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    store.append(_record("a.png", "ok", "ok"))
    out = store.export_csv(tmp_path / "out.csv")
    text = out.read_text(encoding="utf-8")
    assert "image_path" in text.splitlines()[0]
    assert "a.png" in text

"""README results-block generator.

The README's headline table is generated from a run's ``results.json``, and CI
re-checks it. That check has to name *which* run it is checking against --
see :func:`declared_source`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "update_readme_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("update_readme_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _payload(mean_img=85.41, mean_pix=96.44, pro=86.13):
    from spade.metrics import REFERENCE_IMAGE_ROCAUC, REFERENCE_PIXEL_ROCAUC

    summary = {"mean_image_rocauc": mean_img, "mean_pixel_rocauc": mean_pix, "wall_clock_s": 1200.0}
    if pro is not None:
        summary["mean_pixel_pro"] = pro
    return {
        "run_name": "t",
        "config": {"top_k": 5, "backbone": "wide_resnet50_2", "resolved_device": "cpu"},
        "summary": summary,
        "per_category": [
            {
                "category": c,
                "image_rocauc": REFERENCE_IMAGE_ROCAUC[c],
                "pixel_rocauc": REFERENCE_PIXEL_ROCAUC[c],
                **({"pixel_pro": 80.0} if pro is not None else {}),
            }
            for c in REFERENCE_IMAGE_ROCAUC
        ],
        "environment": {"torch": "2.13.0", "platform": "test"},
    }


def test_declared_source_reads_the_marker():
    text = "intro\n<!-- source: artifacts/runs/full-mvtec-k5/results.json -->\nrest"
    assert mod.declared_source(text) == REPO_ROOT / "artifacts/runs/full-mvtec-k5/results.json"


def test_declared_source_is_none_without_a_marker():
    assert mod.declared_source("no marker here") is None


def test_declared_source_tolerates_whitespace():
    assert mod.declared_source("<!--   source:   a/b.json   -->") == REPO_ROOT / "a/b.json"


def test_source_marker_roundtrips():
    path = REPO_ROOT / "artifacts" / "runs" / "x" / "results.json"
    assert mod.declared_source(mod.source_marker(path)) == path


def test_render_includes_pro_column_only_when_present():
    with_pro = mod.render(_payload(pro=86.13))
    without_pro = mod.render(_payload(pro=None))
    assert "PRO" in with_pro and "86.13" in with_pro
    assert "PRO" not in without_pro


def test_render_marks_pro_as_having_no_baseline():
    """A delta against a baseline that never published PRO would be fiction."""
    text = mod.render(_payload(pro=86.13))
    assert "—" in text


def test_check_uses_the_declared_run_not_the_newest(tmp_path, monkeypatch, capsys):
    """Regression: two runs on disk must not make CI check the wrong one.

    ``fallback_results`` picks the most recently modified results.json anywhere
    under artifacts/. Once an ablation run exists it is newer than the canonical
    one, so a marker-less check would compare the README against the ablation
    and report a correct README as stale.
    """
    canonical = tmp_path / "runs" / "canonical" / "results.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(json.dumps(_payload(mean_pix=96.44)), encoding="utf-8")

    ablation = tmp_path / "runs" / "ablation" / "results.json"
    ablation.parent.mkdir(parents=True)
    ablation.write_text(json.dumps(_payload(mean_pix=96.40)), encoding="utf-8")
    # Make the ablation unambiguously newer.
    import os

    os.utime(ablation, (10**9, 10**9 + 1000))

    readme = tmp_path / "README.md"
    readme.write_text(f"# t\n\n{mod.BEGIN}\nplaceholder\n{mod.END}\n", encoding="utf-8")

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "fallback_results", lambda: ablation)

    # Generate against the canonical run...
    monkeypatch.setattr("sys.argv", ["x", str(canonical), "--readme", str(readme)])
    mod.main()
    text = readme.read_text(encoding="utf-8")
    assert mod.source_marker(canonical) in text

    # ...then a check with no explicit path must still resolve to the canonical
    # run via the marker, not to the newer ablation.
    capsys.readouterr()
    monkeypatch.setattr("sys.argv", ["x", "--readme", str(readme), "--check"])
    mod.main()
    assert "up to date" in capsys.readouterr().out

    # And prove the guard is real: without the marker the fallback would pick
    # the ablation and declare the identical README stale.
    readme.write_text(text.replace(mod.source_marker(canonical) + "\n", ""), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["x", "--readme", str(readme), "--check"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert "stale" in str(exc.value)


def test_check_fails_loudly_when_the_declared_run_is_gone(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"{mod.BEGIN}\n<!-- source: artifacts/runs/deleted/results.json -->\n{mod.END}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["x", "--readme", str(readme), "--check"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert "does not exist" in str(exc.value)


def test_partial_run_is_rejected(tmp_path, monkeypatch):
    payload = _payload()
    payload["per_category"] = payload["per_category"][:3]
    results = tmp_path / "results.json"
    results.write_text(json.dumps(payload), encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(f"{mod.BEGIN}\n{mod.END}\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["x", str(results), "--readme", str(readme)])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert "15-category" in str(exc.value)


def test_missing_markers_are_rejected(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text("no markers", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["x", "--readme", str(readme), "--check"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert "markers" in str(exc.value)


def test_committed_readme_is_in_sync():
    """The real README, checked the way CI checks it."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    source = mod.declared_source(text)
    assert source is not None, "README results block must record its source run"
    assert source.exists(), f"README points at a missing run: {source}"

#!/usr/bin/env python
"""Inject a finished run's numbers into the README between the RESULTS markers.

Keeps the headline table in the README honest: it is generated from
``results.json``, never hand-edited.

    python scripts/update_readme_results.py artifacts/runs/full-mvtec-k5/results.json
    python scripts/update_readme_results.py --check    # CI-friendly: fail if stale
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"


def render(payload: dict) -> str:
    rows = payload["per_category"]
    summary = payload["summary"]
    cfg = payload["config"]
    env = payload.get("environment", {})

    mi, mp = summary["mean_image_rocauc"], summary["mean_pixel_rocauc"]
    pro = summary.get("mean_pixel_pro")
    pro_head = " PRO |" if pro is not None else ""
    pro_sep = " ---: |" if pro is not None else ""
    lines = [
        f"All 15 MVTec AD categories, `K={cfg['top_k']}`, `{cfg['backbone']}` "
        "(ImageNet **IMAGENET1K_V1** weights, frozen throughout):",
        "",
        f"| | Image ROC-AUC | Pixel ROC-AUC |{pro_head}",
        f"| --- | ---: | ---: |{pro_sep}",
        f"| **Mean over 15 categories** | **{mi:.2f} %** | **{mp:.2f} %** |"
        + (f" **{pro:.2f} %** |" if pro is not None else ""),
        "",
    ]
    if pro is not None:
        lines += [
            "PRO (per-region overlap, integrated to FPR <= 0.3) weights every defect region "
            "equally, unlike pixel ROC-AUC which large defects dominate: missing a small "
            "scratch costs as much as missing a large one.",
            "",
        ]
    lines += [
        "<details>",
        "<summary>Per-category detail (click to expand)</summary>",
        "",
        "| category | image ROC-AUC | pixel ROC-AUC |" + (" PRO |" if pro is not None else ""),
        "| --- | ---: | ---: |" + (" ---: |" if pro is not None else ""),
    ]
    for r in rows:
        row_pro = r.get("pixel_pro")
        lines.append(
            f"| {r['category']} | {r['image_rocauc']:.2f} | {r['pixel_rocauc']:.2f} |"
            + (f" {'—' if row_pro is None else f'{row_pro:.2f}'} |" if pro is not None else "")
        )
    lines += [
        f"| **Mean** | **{mi:.2f}** | **{mp:.2f}** |"
        + (f" **{pro:.2f}** |" if pro is not None else ""),
        "",
        "</details>",
        "",
        f"> Environment: {env.get('platform', 'n/a')}, PyTorch {env.get('torch', 'n/a')}, "
        f"device `{cfg.get('resolved_device', 'n/a')}`, "
        f"{summary.get('wall_clock_s', 0) / 60:.1f} min end to end.",
        "> The method neither trains nor samples, so repeated runs in the same environment "
        "are bit-for-bit identical.",
        "",
        "`grid` scores 47 % at image level -- below chance. That is a known limit of "
        "global-descriptor retrieval on a regular texture, not a bug: localisation on the "
        "same category is fine (98.35 % pixel, 86.39 % PRO). See "
        "[docs/method.md](docs/method.md#6-expected-variation).",
    ]
    return "\n".join(lines)


def source_marker(results_path: Path) -> str:
    rel = results_path.resolve().relative_to(REPO_ROOT).as_posix()
    return f"<!-- source: {rel} -->"


def declared_source(text: str) -> Path | None:
    """The run the README block was generated from, as recorded in the block.

    Without this, ``--check`` had to guess -- it picked the most recently
    modified ``results.json`` anywhere under ``artifacts/``. The moment a second
    run exists (say an ablation), the guess silently switches to the wrong run
    and CI fails with "README is stale" while the README is in fact correct.
    """
    match = re.search(r"<!--\s*source:\s*(\S+?)\s*-->", text)
    return REPO_ROOT / match.group(1) if match else None


def fallback_results() -> Path:
    """Only used to bootstrap a README that has no source marker yet."""
    candidates = sorted(
        (REPO_ROOT / "artifacts").glob("**/results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit("no results.json found under artifacts/ -- run `make eval` first")
    return candidates[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="?", default=None, help="path to results.json")
    p.add_argument("--readme", default=str(REPO_ROOT / "README.md"))
    p.add_argument("--check", action="store_true", help="exit 1 if the README is out of date")
    args = p.parse_args()

    readme_path = Path(args.readme)
    text = readme_path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{readme_path} is missing the {BEGIN} / {END} markers")

    if args.results:
        results_path = Path(args.results)
    else:
        results_path = declared_source(text) or fallback_results()
    if not results_path.exists():
        raise SystemExit(
            f"{readme_path.name} points at {results_path}, which does not exist. "
            "Re-run the pipeline, or regenerate against a run that does."
        )

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if len(payload["per_category"]) != 15:
        raise SystemExit(
            f"{results_path} covers {len(payload['per_category'])} categories; "
            "the README table needs a full 15-category run"
        )

    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    updated = f"{head}{BEGIN}\n{source_marker(results_path)}\n{render(payload)}\n{END}{tail}"

    if args.check:
        if updated != text:
            raise SystemExit("README results block is stale; run scripts/update_readme_results.py")
        print("README results block is up to date")
        return

    readme_path.write_text(updated, encoding="utf-8")
    print(f"updated {readme_path} from {results_path}")


if __name__ == "__main__":
    main()

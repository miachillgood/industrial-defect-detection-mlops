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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spade.metrics import (  # noqa: E402
    PAPER_MEAN_IMAGE_ROCAUC,
    PAPER_MEAN_PIXEL_ROCAUC,
    REFERENCE_IMAGE_ROCAUC,
    REFERENCE_MEAN_IMAGE_ROCAUC,
    REFERENCE_MEAN_PIXEL_ROCAUC,
    REFERENCE_PIXEL_ROCAUC,
)

BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"


def render(payload: dict) -> str:
    rows = payload["per_category"]
    summary = payload["summary"]
    cfg = payload["config"]
    env = payload.get("environment", {})

    mi, mp = summary["mean_image_rocauc"], summary["mean_pixel_rocauc"]
    pro = summary.get("mean_pixel_pro")
    # PRO has no public baseline to compare against, so its column stays blank
    # in the comparison table rather than inviting a bogus delta.
    pro_head = " PRO |" if pro is not None else ""
    pro_sep = " ---: |" if pro is not None else ""
    lines = [
        f"All 15 MVTec AD categories, `K={cfg['top_k']}`, `{cfg['backbone']}` "
        "(ImageNet **IMAGENET1K_V1** weights, frozen throughout):",
        "",
        f"| | Image ROC-AUC | Pixel ROC-AUC |{pro_head}",
        f"| --- | ---: | ---: |{pro_sep}",
        f"| **This project (K={cfg['top_k']})** | **{mi:.2f} %** | **{mp:.2f} %** |"
        + (f" **{pro:.2f} %** |" if pro is not None else ""),
        f"| Public baseline `byungjae89/SPADE-pytorch` (K=5) | {REFERENCE_MEAN_IMAGE_ROCAUC} % "
        f"| {REFERENCE_MEAN_PIXEL_ROCAUC} % |" + (" — |" if pro is not None else ""),
        f"| Paper (K=50) | {PAPER_MEAN_IMAGE_ROCAUC} % | {PAPER_MEAN_PIXEL_ROCAUC} % |"
        + (" — |" if pro is not None else ""),
        f"| Delta vs. public baseline | {mi - REFERENCE_MEAN_IMAGE_ROCAUC:+.2f} "
        f"| {mp - REFERENCE_MEAN_PIXEL_ROCAUC:+.2f} |" + (" — |" if pro is not None else ""),
        "",
    ]
    if pro is not None:
        lines += [
            "PRO (per-region overlap, integrated to FPR <= 0.3) weights every defect region "
            "equally, unlike pixel ROC-AUC which large defects dominate. Neither the public "
            "baseline nor the paper reports it, so there is no reference column.",
            "",
        ]
    lines += [
        "<details>",
        "<summary>Per-category detail (click to expand)</summary>",
        "",
        "| category | image (baseline) | image (ours) | Δ | pixel (baseline) | pixel (ours) | Δ |"
        + (" PRO |" if pro is not None else ""),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |" + (" ---: |" if pro is not None else ""),
    ]
    for r in rows:
        c = r["category"]
        ri, rp = REFERENCE_IMAGE_ROCAUC[c], REFERENCE_PIXEL_ROCAUC[c]
        row_pro = r.get("pixel_pro")
        lines.append(
            f"| {c} | {ri} | {r['image_rocauc']:.2f} | {r['image_rocauc'] - ri:+.2f} "
            f"| {rp} | {r['pixel_rocauc']:.2f} | {r['pixel_rocauc'] - rp:+.2f} |"
            + (f" {'—' if row_pro is None else f'{row_pro:.2f}'} |" if pro is not None else "")
        )
    lines += [
        f"| **Mean** | **{REFERENCE_MEAN_IMAGE_ROCAUC}** | **{mi:.2f}** "
        f"| **{mi - REFERENCE_MEAN_IMAGE_ROCAUC:+.2f}** | **{REFERENCE_MEAN_PIXEL_ROCAUC}** "
        f"| **{mp:.2f}** | **{mp - REFERENCE_MEAN_PIXEL_ROCAUC:+.2f}** |"
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
        "`grid` scores 47 % at image level -- below chance. That is not a bug but a known "
        "weakness of the method; the public baseline reports 47.3 % too. See "
        "[docs/method.md](docs/method.md#6-已知的正常波动).",
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

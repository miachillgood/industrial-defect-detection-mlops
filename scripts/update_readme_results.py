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
    lines = [
        f"MVTec AD 全部 15 类跑通，`K={cfg['top_k']}`、`{cfg['backbone']}`（ImageNet **IMAGENET1K_V1** 权重，全程冻结）：",
        "",
        "| | 图像级 ROC-AUC | 像素级 ROC-AUC |",
        "| --- | ---: | ---: |",
        f"| **本项目（K={cfg['top_k']}）** | **{mi:.2f} %** | **{mp:.2f} %** |",
        f"| 公开基准 `byungjae89/SPADE-pytorch`（K=5） | {REFERENCE_MEAN_IMAGE_ROCAUC} % | {REFERENCE_MEAN_PIXEL_ROCAUC} % |",
        f"| 论文基准（K=50） | {PAPER_MEAN_IMAGE_ROCAUC} % | {PAPER_MEAN_PIXEL_ROCAUC} % |",
        f"| 与公开基准之差 | {mi - REFERENCE_MEAN_IMAGE_ROCAUC:+.2f} | {mp - REFERENCE_MEAN_PIXEL_ROCAUC:+.2f} |",
        "",
        "<details>",
        "<summary>逐类明细（点开）</summary>",
        "",
        "| 类别 | 图像级 基准 | 图像级 本项目 | Δ | 像素级 基准 | 像素级 本项目 | Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        c = r["category"]
        ri, rp = REFERENCE_IMAGE_ROCAUC[c], REFERENCE_PIXEL_ROCAUC[c]
        lines.append(
            f"| {c} | {ri} | {r['image_rocauc']:.2f} | {r['image_rocauc'] - ri:+.2f} "
            f"| {rp} | {r['pixel_rocauc']:.2f} | {r['pixel_rocauc'] - rp:+.2f} |"
        )
    lines += [
        f"| **均值** | **{REFERENCE_MEAN_IMAGE_ROCAUC}** | **{mi:.2f}** "
        f"| **{mi - REFERENCE_MEAN_IMAGE_ROCAUC:+.2f}** | **{REFERENCE_MEAN_PIXEL_ROCAUC}** "
        f"| **{mp:.2f}** | **{mp - REFERENCE_MEAN_PIXEL_ROCAUC:+.2f}** |",
        "",
        "</details>",
        "",
        f"> 运行环境：{env.get('platform', 'n/a')}，PyTorch {env.get('torch', 'n/a')}，"
        f"设备 `{cfg.get('resolved_device', 'n/a')}`，全程耗时 {summary.get('wall_clock_s', 0) / 60:.1f} 分钟。",
        "> 本方法不训练也不采样，同一环境下重复运行结果完全一致。",
        "",
        "`grid` 的图像级 47 % 低于随机——这不是 bug，是这套方法的已知短板，公开基准同样是 47.3 %；"
        "详见 [docs/method.md](docs/method.md#6-已知的正常波动)。",
    ]
    return "\n".join(lines)


def latest_results() -> Path:
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

    results_path = Path(args.results) if args.results else latest_results()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if len(payload["per_category"]) != 15:
        raise SystemExit(
            f"{results_path} covers {len(payload['per_category'])} categories; "
            "the README table needs a full 15-category run"
        )

    readme_path = Path(args.readme)
    text = readme_path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{readme_path} is missing the {BEGIN} / {END} markers")

    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    updated = f"{head}{BEGIN}\n{render(payload)}\n{END}{tail}"

    if args.check:
        if updated != text:
            raise SystemExit("README results block is stale; run scripts/update_readme_results.py")
        print("README results block is up to date")
        return

    readme_path.write_text(updated, encoding="utf-8")
    print(f"updated {readme_path} from {results_path}")


if __name__ == "__main__":
    main()

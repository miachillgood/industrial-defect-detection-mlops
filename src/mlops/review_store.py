"""Append-only store for human review decisions.

The Streamlit tool writes one JSON object per line. Append-only means two
reviewers can work at once without clobbering each other, and the full history
survives -- a corrected decision is a new record, not an overwrite.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_STORE = Path("artifacts/reviews/reviews.jsonl")

# What a human can say about a sample.
VERDICTS = ("ok", "defect", "unsure")


@dataclass
class ReviewRecord:
    image_path: str
    category: str
    human_verdict: str
    model_score: float
    model_threshold: float
    model_verdict: str
    reviewer: str = "anonymous"
    defect_type: str | None = None
    notes: str = ""
    ground_truth: str | None = None  # filled in when reviewing the MVTec test split
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def __post_init__(self) -> None:
        if self.human_verdict not in VERDICTS:
            raise ValueError(f"human_verdict must be one of {VERDICTS}, got {self.human_verdict!r}")

    @property
    def agrees_with_model(self) -> bool:
        return self.human_verdict == self.model_verdict


class ReviewStore:
    def __init__(self, path: str | os.PathLike = DEFAULT_STORE) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ReviewRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn write at the tail
        return rows

    def latest_per_image(self) -> dict[str, dict]:
        """Last decision wins, keyed by image path."""
        out: dict[str, dict] = {}
        for row in self.load():
            out[row["image_path"]] = row
        return out

    def stats(self) -> dict:
        rows = list(self.latest_per_image().values())
        if not rows:
            return {"n_reviewed": 0}

        agree = sum(1 for r in rows if r["human_verdict"] == r["model_verdict"])
        # Confusion of the model against the human verdict, ignoring "unsure".
        decided = [r for r in rows if r["human_verdict"] in ("ok", "defect")]
        tp = sum(1 for r in decided if r["human_verdict"] == "defect" and r["model_verdict"] == "defect")
        fp = sum(1 for r in decided if r["human_verdict"] == "ok" and r["model_verdict"] == "defect")
        fn = sum(1 for r in decided if r["human_verdict"] == "defect" and r["model_verdict"] == "ok")
        tn = sum(1 for r in decided if r["human_verdict"] == "ok" and r["model_verdict"] == "ok")

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "n_reviewed": len(rows),
            "n_agree": agree,
            "agreement_rate": round(agree / len(rows), 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_unsure": sum(1 for r in rows if r["human_verdict"] == "unsure"),
            "reviewers": sorted({r.get("reviewer", "anonymous") for r in rows}),
        }

    def export_csv(self, path: str | os.PathLike) -> Path:
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = list(self.latest_per_image().values())
        fields = [
            "timestamp", "reviewer", "category", "image_path", "human_verdict",
            "model_verdict", "model_score", "model_threshold", "defect_type",
            "ground_truth", "notes",
        ]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

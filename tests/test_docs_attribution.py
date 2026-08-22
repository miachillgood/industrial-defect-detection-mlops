"""Credit guards.

The algorithm this project implements is published work, so the credit line must
survive edits, and two specific mistakes must never creep into the docs:

* calling ``byungjae89/SPADE-pytorch`` the *official* SPADE repository -- there
  is no public repository from the paper's authors;
* confusing this SPADE with ``NVlabs/SPADE`` (spatially-adaptive normalisation,
  an image-synthesis method that shares the acronym).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
DOC_FILES = [README, *sorted((REPO_ROOT / "docs").glob("*.md"))]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_exists():
    assert README.is_file(), "README.md is the entry point for this project"


@pytest.mark.parametrize(
    "needle",
    [
        "byungjae89/SPADE-pytorch",
        "2005.02357",
        "Cohen",
        "Hoshen",
        "MVTec",
    ],
)
def test_readme_credits_the_sources(needle):
    assert needle in _text(README), f"README must mention {needle!r}"


def test_readme_warns_about_the_nvlabs_namesake():
    text = _text(README)
    assert "NVlabs/SPADE" in text
    assert re.search(r"[Ss]patially[- ][Aa]daptive [Nn]ormali[sz]ation", text), (
        "spell out what NVlabs/SPADE actually is, so the confusion is unmistakable"
    )


def test_readme_documents_the_k_tradeoff():
    """K=5 is a chosen default, not an inherited one -- the README must show
    the measurement behind the choice, which means quoting both settings."""
    text = _text(README)
    assert "K=5" in text and "K=50" in text


# Phrases that would misattribute the third-party implementation.
FORBIDDEN_CLAIMS = [
    r"official\s+SPADE\s+(repo|repository|implementation|code)",
    r"SPADE\s*官方\s*(仓库|实现|代码)",
    r"官方\s*SPADE",
    r"authors['’]?\s+official\s+(repo|repository|code)",
]

# The docs *must* be able to quote a forbidden phrase in order to warn against
# it. A nearby negation cue is what separates "this is the official repo" from
# "do not call this the official repo".
NEGATION_CUES = (
    "不能", "不要", "不是", "并非", "而非", "误", "错", "禁止", "避免",
    "not ", "never", "must not", "do not", "don't", "rather than", "instead of",
    "incorrect", "wrong", "mistake",
)


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_no_doc_claims_the_reference_repo_is_official(doc):
    text = _text(doc)
    for pattern in FORBIDDEN_CLAIMS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            window = text[max(0, match.start() - 120) : match.end() + 120].lower()
            assert any(cue in window for cue in NEGATION_CUES), (
                f"{doc.name} asserts the third-party implementation is official: "
                f"{match.group(0)!r} (no negation nearby). There is no public "
                "repository from the SPADE paper's authors."
            )


def test_the_attribution_guard_actually_catches_a_bare_claim(tmp_path):
    """Guard the guard: an unqualified claim must still fail."""
    bad = "See the official SPADE repository for the code."
    hits = [
        m
        for pattern in FORBIDDEN_CLAIMS
        for m in re.finditer(pattern, bad, flags=re.IGNORECASE)
    ]
    assert hits, "the forbidden-claim patterns no longer match an unqualified claim"
    window = bad.lower()
    assert not any(cue in window for cue in NEGATION_CUES)


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_docs_do_not_link_the_wrong_spade_as_the_source(doc):
    """A bare link to NVlabs/SPADE is fine only in the disambiguation context."""
    text = _text(doc)
    for match in re.finditer(r"github\.com/NVlabs/SPADE", text):
        window = text[max(0, match.start() - 260) : match.end() + 260].lower()
        assert any(
            hint in window
            for hint in ("not", "different", "unrelated", "confus", "不是", "无关", "注意")
        ), f"{doc.name} links NVlabs/SPADE without flagging that it is a different method"


def test_readme_quotes_this_projects_own_headline_numbers():
    """Guards against the headline drifting from artifacts/runs/full-mvtec-k5.

    Asserted at full precision on purpose: `85.4` would also match `85.41`, so
    a looser check could pass while quoting somebody else's figure.
    """
    text = _text(README)
    assert "85.41" in text and "96.44" in text

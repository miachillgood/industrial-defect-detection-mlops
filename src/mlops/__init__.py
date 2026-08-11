"""MLOps layer built on top of the detection core.

Nothing here is part of the SPADE method itself -- the reference implementation
(``byungjae89/SPADE-pytorch``) covers only detection and evaluation. This is the
engineering wrapped around it.

* :mod:`mlops.tracking`     -- MLflow experiment tracking (fail-soft)
* :mod:`mlops.review_store` -- append-only store for human review decisions
"""

__all__ = ["review_store", "tracking"]

"""Unsupervised industrial defect detection: feature pyramid + kNN retrieval.

Implements *Sub-Image Anomaly Detection with Deep Pyramid Correspondences*
(Cohen & Hoshen, arXiv:2005.02357) following the third-party PyTorch
implementation `byungjae89/SPADE-pytorch`.

Not to be confused with NVlabs/SPADE (spatially-adaptive normalisation for image
synthesis) -- an unrelated method that happens to share the acronym.
"""

from .config import SpadeConfig, pick_device
from .data import CLASS_NAMES, MVTecDataset
from .features import PyramidFeatureExtractor, extract_features
from .model import SPADE, SpadeOutput

__all__ = [
    "SpadeConfig",
    "pick_device",
    "CLASS_NAMES",
    "MVTecDataset",
    "PyramidFeatureExtractor",
    "extract_features",
    "SPADE",
    "SpadeOutput",
]

__version__ = "0.1.0"

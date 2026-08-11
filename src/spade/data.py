"""MVTec AD dataset access.

Layout expected under ``root``::

    <root>/<category>/train/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/*_mask.png

The pre-processing chain is kept identical to the reference implementation:
resize the short side to 256 with **LANCZOS** (old torchvision's
``Image.ANTIALIAS``), centre-crop to 224, then ImageNet normalisation. Masks use
nearest-neighbour resizing so labels stay binary.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

CLASS_NAMES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

TEXTURE_CLASSES = frozenset({"carpet", "grid", "leather", "tile", "wood"})

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(resize: int = 256, cropsize: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(resize, interpolation=InterpolationMode.LANCZOS),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_mask_transform(resize: int = 256, cropsize: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(resize, interpolation=InterpolationMode.NEAREST),
            T.CenterCrop(cropsize),
            T.ToTensor(),
        ]
    )


def denormalize(x) -> np.ndarray:  # noqa: F821 - numpy imported lazily
    """Undo ImageNet normalisation; returns an HWC uint8 array."""
    import numpy as np

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return (((x.transpose(1, 2, 0) * std) + mean) * 255.0).clip(0, 255).astype("uint8")


def resolve_root(root: str | os.PathLike) -> Path:
    """Accept either the dataset directory or its parent, and cope with the
    extra nesting level some mirrors ship."""
    root = Path(root)
    candidates = [
        root,
        root / "mvtec_anomaly_detection",
        root / "mvtech_anomaly_detection",
    ]
    for cand in candidates:
        if (cand / "bottle" / "train" / "good").is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find an MVTec AD tree under {root!s}. "
        "Expected <root>/bottle/train/good to exist. "
        "Run `python scripts/prepare_data.py --help` for download instructions."
    )


class MVTecDataset(Dataset):
    """One MVTec AD category, either the train split (all-good) or the test split."""

    def __init__(
        self,
        root: str | os.PathLike,
        category: str = "bottle",
        is_train: bool = True,
        resize: int = 256,
        cropsize: int = 224,
    ) -> None:
        if category not in CLASS_NAMES:
            raise ValueError(f"category {category!r} must be one of {CLASS_NAMES}")
        self.root = resolve_root(root)
        self.category = category
        self.is_train = is_train
        self.cropsize = cropsize

        self.image_paths, self.labels, self.mask_paths = self._load_index()
        self.transform_x = build_image_transform(resize, cropsize)
        self.transform_mask = build_mask_transform(resize, cropsize)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path, label, mask_path = (
            self.image_paths[idx],
            self.labels[idx],
            self.mask_paths[idx],
        )
        x = self.transform_x(Image.open(path).convert("RGB"))
        if label == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = self.transform_mask(Image.open(mask_path))
            # Ground-truth PNGs are 0/255; ToTensor maps them to 0.0/1.0 already,
            # but binarise defensively in case a mirror stores them differently.
            mask = (mask > 0.5).float()
        return x, label, mask

    def _load_index(self) -> tuple[list[str], list[int], list[str | None]]:
        phase = "train" if self.is_train else "test"
        img_dir = self.root / self.category / phase
        gt_dir = self.root / self.category / "ground_truth"

        images: list[str] = []
        labels: list[int] = []
        masks: list[str | None] = []

        for defect_type in sorted(p.name for p in img_dir.iterdir() if p.is_dir()):
            files = sorted((img_dir / defect_type).glob("*.png"))
            if not files:
                continue
            images.extend(str(f) for f in files)
            if defect_type == "good":
                labels.extend([0] * len(files))
                masks.extend([None] * len(files))
            else:
                labels.extend([1] * len(files))
                masks.extend(str(gt_dir / defect_type / f"{f.stem}_mask.png") for f in files)

        if not images:
            raise FileNotFoundError(f"no PNGs found under {img_dir}")
        return images, labels, masks

    def summary(self) -> dict:
        return {
            "category": self.category,
            "split": "train" if self.is_train else "test",
            "n_images": len(self.image_paths),
            "n_anomalous": int(sum(self.labels)),
            "n_good": int(len(self.labels) - sum(self.labels)),
        }

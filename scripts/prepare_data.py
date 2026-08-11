#!/usr/bin/env python
"""Fetch and unpack the MVTec AD dataset into ``data/mvtec_anomaly_detection``.

MVTec AD is released for non-commercial research use under CC BY-NC-SA 4.0.
Please read and accept the licence on the official page before using it:
https://www.mvtec.com/company/research/datasets/mvtec-ad

The URL hard-coded in the upstream reference implementation
(``ftp://guest:...@ftp.softronics.ch/...``) no longer accepts logins, so this
script takes a mirror URL and verifies the extracted tree instead.

Examples
--------
    python scripts/prepare_data.py --check-only
    python scripts/prepare_data.py --archive ~/Downloads/mvtec_anomaly_detection.tar.xz
    python scripts/prepare_data.py --url <mirror-url>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spade.data import CLASS_NAMES  # noqa: E402

# A community mirror of the official archive. Prefer downloading from mvtec.com
# directly if you can; this exists because the upstream FTP mirror is dead.
DEFAULT_URL = (
    "https://huggingface.co/datasets/hdtech/mvtech_anomaly_detection/"
    "resolve/main/mvtech_anomaly_detection.zip"
)

EXPECTED_TRAIN_COUNTS = {
    "bottle": 209, "cable": 224, "capsule": 219, "carpet": 280, "grid": 264,
    "hazelnut": 391, "leather": 245, "metal_nut": 220, "pill": 267, "screw": 320,
    "tile": 230, "toothbrush": 60, "transistor": 213, "wood": 247, "zipper": 240,
}


def download(url: str, dest: Path) -> Path:
    import urllib.request

    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[prepare] archive already present: {dest}")
        return dest

    class _Bar(tqdm):
        def update_to(self, blocks=1, block_size=1, total=None):
            if total is not None:
                self.total = total
            self.update(blocks * block_size - self.n)

    print(f"[prepare] downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _Bar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as bar:
        urllib.request.urlretrieve(url, tmp, reporthook=bar.update_to)
    tmp.rename(dest)
    return dest


def extract(archive: Path, target: Path) -> Path:
    staging = target.parent / "_extract_tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"[prepare] extracting {archive.name} -> {target}")
    if archive.suffix == ".zip" or zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
    else:
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(staging)

    # Mirrors differ in how many directory levels they wrap the categories in.
    root = _find_dataset_root(staging)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root), str(target))
    shutil.rmtree(staging, ignore_errors=True)
    return target


def _find_dataset_root(base: Path) -> Path:
    for path in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        if (path / "bottle" / "train" / "good").is_dir():
            return path
    raise RuntimeError(f"no MVTec AD tree (a directory containing bottle/train/good) under {base}")


def verify(root: Path, strict: bool = False) -> bool:
    ok = True
    print(f"[prepare] verifying {root}")
    for category in CLASS_NAMES:
        train_dir = root / category / "train" / "good"
        test_dir = root / category / "test"
        gt_dir = root / category / "ground_truth"
        if not train_dir.is_dir() or not test_dir.is_dir() or not gt_dir.is_dir():
            print(f"  MISSING  {category}")
            ok = False
            continue
        n_train = len(list(train_dir.glob("*.png")))
        n_test = len(list(test_dir.rglob("*.png")))
        n_masks = len(list(gt_dir.rglob("*_mask.png")))
        expected = EXPECTED_TRAIN_COUNTS[category]
        flag = "" if n_train == expected else f"  <-- expected {expected} train images"
        if n_train != expected:
            ok = False
        print(f"  {category:<12} train={n_train:<4} test={n_test:<4} masks={n_masks:<4}{flag}")
    print(f"[prepare] {'OK' if ok else 'INCOMPLETE'}")
    if strict and not ok:
        raise SystemExit(1)
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    p.add_argument("--url", default=DEFAULT_URL, help="mirror to download from")
    p.add_argument("--archive", default=None, help="use an archive you already downloaded")
    p.add_argument("--check-only", action="store_true", help="only verify an existing tree")
    p.add_argument("--strict", action="store_true", help="exit non-zero if verification fails")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    target = data_dir / "mvtec_anomaly_detection"

    if args.check_only:
        if not target.exists():
            print(f"[prepare] {target} does not exist")
            raise SystemExit(1 if args.strict else 0)
        verify(target, strict=args.strict)
        return

    if not target.exists():
        archive = Path(args.archive) if args.archive else download(args.url, data_dir / "mvtec_ad.zip")
        extract(archive, target)
    else:
        print(f"[prepare] dataset already extracted at {target}")

    verify(target, strict=args.strict)


if __name__ == "__main__":
    main()

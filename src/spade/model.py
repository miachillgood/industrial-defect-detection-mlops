"""SPADE: Sub-Image Anomaly Detection with Deep Pyramid Correspondences.

Implementation of the algorithm described in Cohen & Hoshen, arXiv:2005.02357,
following the third-party PyTorch implementation ``byungjae89/SPADE-pytorch``.

Two stages, neither of which involves any training:

1. **Image level.** Every training image is summarised by the 2048-d global
   average-pooled activation. A test image's score is the mean Euclidean
   distance to its ``K`` nearest training images.
2. **Pixel level.** For the same ``K`` neighbours, build a gallery of feature
   vectors over *all* spatial positions at layer1/2/3. Each test position is
   scored by its distance to the closest gallery vector; the per-layer maps are
   upsampled to input resolution, averaged, and Gaussian-smoothed.

Fidelity notes
--------------
* ``K=5`` by default. The paper uses ``K=50``; the reference implementation we
  public baseline uses ``K=5`` and reports 85.4 / 96.4.
* The reference calls ``torch.pairwise_distance`` on broadcast 4-D tensors. Under
  PyTorch 1.x that reduced over ``dim=1`` (the channel axis), which is the
  intended semantics. PyTorch >= 2.0 reduces over the **last** axis instead, so
  the original line silently computes the wrong quantity on a modern install.
  We reduce over the channel axis explicitly.
* The reference adds ``eps=1e-6`` inside the norm, i.e. ``||x1 - x2 + eps||``.
  That is identical to shifting the gallery by ``eps``, which is what we do so
  the distance can be evaluated with a fast matmul-based ``cdist``.
* The reference iterates ``range(gallery_size // 100)``, which silently drops the
  final ``gallery_size % 100`` rows (<0.6 % of the gallery). We use the whole
  gallery; set ``drop_gallery_remainder=True`` to mirror that quirk exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

EPS = 1e-6


@dataclass
class SpadeOutput:
    """Per-test-image results for one category."""

    image_scores: np.ndarray  # (N,)
    score_maps: np.ndarray  # (N, cropsize, cropsize)
    neighbour_indices: np.ndarray  # (N, K) indices into the training bank


def _pairwise_min_distance(
    query: torch.Tensor,
    gallery: torch.Tensor,
    chunk: int = 4096,
) -> torch.Tensor:
    """Smallest Euclidean distance from each query row to any gallery row.

    ``query``   -- (Q, C)
    ``gallery`` -- (G, C)
    returns     -- (Q,)

    Chunked over the gallery so peak memory is ``Q * chunk`` floats regardless of
    how large the gallery is. Chunking changes nothing about the result.
    """
    best = torch.full((query.shape[0],), float("inf"), device=query.device, dtype=query.dtype)
    for start in range(0, gallery.shape[0], chunk):
        block = gallery[start : start + chunk]
        dist = torch.cdist(query, block)  # (Q, block)
        best = torch.minimum(best, dist.min(dim=1).values)
    return best


class SPADE:
    """Memory-bank anomaly detector. ``fit`` only stores features."""

    def __init__(
        self,
        top_k: int = 5,
        layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
        cropsize: int = 224,
        gaussian_sigma: float = 4.0,
        gallery_chunk: int = 4096,
        device: str = "cpu",
        drop_gallery_remainder: bool = False,
    ) -> None:
        self.top_k = top_k
        self.layers = tuple(layers)
        self.cropsize = cropsize
        self.gaussian_sigma = gaussian_sigma
        self.gallery_chunk = gallery_chunk
        self.device = device
        self.drop_gallery_remainder = drop_gallery_remainder
        self.bank: dict[str, torch.Tensor] | None = None

    # -------------------------------------------------------------------- fit
    def fit(self, train_features: dict[str, torch.Tensor]) -> SPADE:
        missing = [k for k in (*self.layers, "avgpool") if k not in train_features]
        if missing:
            raise KeyError(f"training features are missing {missing}")
        self.bank = {k: v for k, v in train_features.items()}
        return self

    @property
    def n_train(self) -> int:
        if self.bank is None:
            raise RuntimeError("call fit() first")
        return self.bank["avgpool"].shape[0]

    # ---------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(
        self,
        test_features: dict[str, torch.Tensor],
        progress_desc: str | None = None,
        exclude_self: bool = False,
    ) -> SpadeOutput:
        """Score a batch of images against the memory bank.

        ``exclude_self=True`` masks the diagonal of the distance matrix. Use it
        when scoring the *training* images themselves (leave-one-out), which is
        how we calibrate decision thresholds without touching the test split.
        """
        if self.bank is None:
            raise RuntimeError("call fit() first")

        k = min(self.top_k, self.n_train - (1 if exclude_self else 0))
        dev = torch.device(self.device)

        # --- stage 1: image-level kNN on the global descriptor ---------------
        train_avg = torch.flatten(self.bank["avgpool"], 1).to(dev, dtype=torch.float32)
        test_avg = torch.flatten(test_features["avgpool"], 1).to(dev, dtype=torch.float32)
        dist_matrix = torch.cdist(test_avg, train_avg)  # (N_test, N_train)
        if exclude_self:
            if dist_matrix.shape[0] != dist_matrix.shape[1]:
                raise ValueError("exclude_self requires scoring the training set itself")
            dist_matrix = dist_matrix.clone()
            dist_matrix.fill_diagonal_(float("inf"))
        topk_values, topk_indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)
        image_scores = topk_values.mean(dim=1).cpu().numpy()
        topk_indices_cpu = topk_indices.cpu()

        # --- stage 2: pixel-level deep pyramid correspondence ----------------
        n_test = test_avg.shape[0]
        iterator = range(n_test)
        if progress_desc:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=progress_desc, leave=False)

        score_maps = np.empty((n_test, self.cropsize, self.cropsize), dtype=np.float32)
        for t_idx in iterator:
            neighbours = topk_indices_cpu[t_idx]
            per_layer = []
            for layer in self.layers:
                gallery = self.bank[layer][neighbours]  # (K, C, H, W)
                query_map = test_features[layer][t_idx]  # (C, H, W)
                c, h, w = query_map.shape

                gallery = gallery.to(dev, dtype=torch.float32)
                gallery = gallery.permute(0, 2, 3, 1).reshape(-1, c)  # (K*H*W, C)
                if self.drop_gallery_remainder:
                    gallery = gallery[: (gallery.shape[0] // 100) * 100]
                # ||x1 - x2 + eps|| == ||(x1 + eps) - x2||
                gallery = gallery + EPS

                query = query_map.to(dev, dtype=torch.float32).reshape(c, h * w).t()  # (H*W, C)
                dist = _pairwise_min_distance(query, gallery, chunk=self.gallery_chunk)

                upsampled = F.interpolate(
                    dist.reshape(1, 1, h, w),
                    size=self.cropsize,
                    mode="bilinear",
                    align_corners=False,
                )
                per_layer.append(upsampled)

            # average the pyramid, then smooth
            fused = torch.cat(per_layer, dim=0).mean(dim=0).squeeze().cpu().numpy()
            score_maps[t_idx] = gaussian_filter(fused, sigma=self.gaussian_sigma)

        return SpadeOutput(
            image_scores=image_scores,
            score_maps=score_maps,
            neighbour_indices=topk_indices_cpu.numpy(),
        )

    # ------------------------------------------------------------ persistence
    def save_bank(self, path, dtype: str = "float16", metadata: dict | None = None) -> None:
        """Persist the memory bank. ``float16`` halves the file for ~no accuracy cost."""
        if self.bank is None:
            raise RuntimeError("call fit() first")
        torch_dtype = getattr(torch, dtype)
        payload = {
            "bank": {k: v.to(torch_dtype) for k, v in self.bank.items()},
            "config": {
                "top_k": self.top_k,
                "layers": list(self.layers),
                "cropsize": self.cropsize,
                "gaussian_sigma": self.gaussian_sigma,
                "bank_dtype": dtype,
            },
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_bank(cls, path, device: str = "cpu", gallery_chunk: int = 4096) -> tuple[SPADE, dict]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        model = cls(
            top_k=cfg["top_k"],
            layers=tuple(cfg["layers"]),
            cropsize=cfg["cropsize"],
            gaussian_sigma=cfg["gaussian_sigma"],
            gallery_chunk=gallery_chunk,
            device=device,
        )
        model.fit({k: v.float() for k, v in payload["bank"].items()})
        return model, payload.get("metadata", {})

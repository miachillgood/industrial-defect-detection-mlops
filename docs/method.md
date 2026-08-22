# Method and implementation detail

This file records how the detection algorithm is implemented, why each configuration choice was made, and where it deviates from the obvious approach and why. Every configuration value that moves a number is written down here, so the figures in the README can be re-derived rather than taken on trust.

---

## 1. Algorithm: two-stage retrieval

The method **trains no parameters at all**. The "model" is a frozen ImageNet classifier plus a bank of features from defect-free training images.

### Stage 1: image-level retrieval

Each image is summarised by the 2048-d global average-pooled Wide-ResNet50-2 descriptor. A test image's anomaly score is the mean Euclidean distance to its **K nearest training images**.

```
score(x) = mean( topK_min ‖ f_avg(x) − f_avg(t_i) ‖₂ ,  t_i ∈ train )
```

### Stage 2: pixel-level deep pyramid correspondence

Taking the same K neighbours from stage 1, the feature vectors at **every spatial position** of those K images are flattened into a single gallery, at `layer1 / layer2 / layer3`. Each test position scores as its distance to the closest gallery vector; the three scales are each bilinearly upsampled to 224×224, averaged, and smoothed with a σ=4 Gaussian.

Resolutions: `layer1` 56×56 (256 channels), `layer2` 28×28 (512), `layer3` 14×14 (1024). At K=5, the `layer1` gallery holds 5×56×56 = 15 680 vectors of dimension 256.

---

## 2. Configuration that matters

Every item below changes the numbers if you get it wrong. §3 expands on the four that are easiest to get wrong silently.

| Item | Setting | Note |
| --- | --- | --- |
| Backbone | `wide_resnet50_2`, `IMAGENET1K_V1` weights | V2 weights shift every number — see §3.2 |
| Hook points | `layer1[-1] / layer2[-1] / layer3[-1] / avgpool` | last bottleneck of each stage |
| Resize | 256, `InterpolationMode.LANCZOS` | PIL's `ANTIALIAS` is LANCZOS, not bilinear; swapping it shifts results systematically |
| Crop | `CenterCrop(224)` | |
| Normalisation | ImageNet mean/std | |
| Mask interpolation | `InterpolationMode.NEAREST` | keeps labels binary |
| K | 5 (`--top-k` to change) | trade-off measured in §7 |
| Image-level distance | `torch.cdist` over flattened avgpool | |
| Image-level score | mean of the top-K distances | |
| Pixel-level scales | layer1 / layer2 / layer3 | |
| Fusion | upsample all three to 224, then average | |
| Smoothing | `gaussian_filter(sigma=4)` | |
| Distance eps | `‖(x₁ + 1e-6) − x₂‖` | equivalent rewrite of `‖x₁ − x₂ + 1e-6‖`, so a matmul-based `cdist` can be used |
| Gallery traversal | full gallery | the obvious `range(G // 100)` loop silently drops the tail — see §3.3 |

---

## 3. Four traps that must be handled explicitly

### 3.1 `torch.pairwise_distance` changed its reduction axis in PyTorch 2.x

The common formulation is this line:

```python
dist_matrix = torch.pairwise_distance(feat_gallery[a:b], test_feat_map)
# feat_gallery : (100, C, 1, 1)
# test_feat_map: (1,   C, H, W)
```

It relies on the norm being taken over **`dim=1` (the channel axis)** after broadcasting, yielding `(100, H, W)` — which is correct on PyTorch 1.x. From PyTorch 2.0, `pairwise_distance` reduces over the **last** axis instead, so the same input yields `(100, C, H)`: a completely different quantity, and no error is raised.

Measured here on PyTorch 2.13:

```
pairwise_distance((4,8,1,1), (1,8,3,5)) -> (4, 8, 3)     # not (4, 3, 5)
```

This project reduces over the channel axis explicitly in `src/spade/model.py`, with a regression test pinning the semantics so nobody "simplifies" it back to the built-in:

```
tests/test_model.py::test_channelwise_reduction_is_the_intended_distance
```

### 3.2 The ImageNet weights must be V1

Early torchvision's `pretrained=True` is equivalent to `IMAGENET1K_V1`. The `IMAGENET1K_V2` weights added later were retrained with an improved recipe; the feature distribution differs and every number moves. `features.py` hard-codes V1 and rejects other backbones rather than silently degrading.

### 3.3 Common implementations drop the tail of the gallery

```python
for d_idx in range(feat_gallery.shape[0] // 100):
    ...feat_gallery[d_idx * 100 : d_idx * 100 + 100]...
```

Integer division discards the last `G % 100` rows. At K=5: `layer1` 15 680 → 15 600 used (80 dropped), `layer2` 3 920 → 3 900 (20 dropped), `layer3` 980 → 900 (80 dropped).

This is clearly unintentional, and this project **uses the full gallery by default**. But "how much does it matter" should not be settled by reasoning, so both modes were run over all 15 categories:

```bash
python -m spade.evaluate --categories all                              # full gallery (default)
python -m spade.evaluate --categories all --drop-gallery-remainder     # replicate the truncation
```

Pixel-level ROC-AUC (%), full gallery vs. truncated:

| category | full gallery | truncated | Δ |
| --- | ---: | ---: | ---: |
| bottle | 97.01 | 96.99 | −0.02 |
| cable | 92.41 | 92.32 | −0.09 |
| capsule | 98.39 | 98.38 | −0.01 |
| carpet | 98.91 | 98.91 | 0.00 |
| grid | 98.35 | 98.32 | −0.03 |
| hazelnut | 98.53 | 98.52 | −0.01 |
| leather | 99.32 | 99.32 | 0.00 |
| metal_nut | 97.14 | 97.06 | −0.08 |
| pill | 94.98 | 94.99 | +0.01 |
| screw | 99.11 | 99.10 | −0.01 |
| tile | 92.88 | 92.83 | −0.05 |
| toothbrush | 98.85 | 98.83 | −0.02 |
| transistor | 86.75 | 86.58 | −0.17 |
| wood | 95.33 | 95.31 | −0.02 |
| zipper | 98.59 | 98.55 | −0.04 |
| **Mean** | **96.44** | **96.40** | **−0.04** |

What the measurement shows:

- Dropping under 0.6 % of the gallery costs 0.04 points on the mean, and the effect is **almost always negative** — as it must be, since a smaller gallery can only leave the nearest-neighbour distance unchanged or make it larger.
- The damage is concentrated, not spread: `transistor` moves 0.17 while five categories move 0.02 or less. Categories whose defects rely on fine spatial correspondence lose the most.
- **Image level is completely unaffected**: the two runs are bit-for-bit identical across all 15 categories (mean 85.41 either way). That follows, because the image-level score comes from the avgpool descriptor and never passes through the gallery.

Both runs are committed: `artifacts/runs/full-mvtec-k5/` (default) and `artifacts/runs/full-mvtec-k5-truncated/` (truncated). `--drop-gallery-remainder` keeps the truncated path available so the cost stays measurable rather than assumed.

### 3.4 The published data download URL is dead

`ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/...` now returns `530 Login denied` (curl exit code 67). `scripts/prepare_data.py` was changed to accept a mirror URL or a local archive, and to verify the per-category image counts after extraction.

---

## 4. Dataset integrity check

`scripts/prepare_data.py --check-only` verifies the training-image count of all 15 categories:

| category | train | category | train | category | train |
| --- | ---: | --- | ---: | --- | ---: |
| bottle | 209 | grid | 264 | screw | 320 |
| cable | 224 | hazelnut | 391 | tile | 230 |
| capsule | 219 | leather | 245 | toothbrush | 60 |
| carpet | 280 | metal_nut | 220 | transistor | 213 |
| | | pill | 267 | wood | 247 |
| | | | | zipper | 240 |

3 629 training images and 1 725 test images in total. A count mismatch means an incomplete mirror, and the script exits non-zero.

---

## 5. Metric definitions

- **Image-level ROC-AUC**: one score per test image, AUC against the good/anomalous label, then the arithmetic mean over the 15 categories.
- **Pixel-level ROC-AUC**: all pixels of *all* test images in a category flattened into one sequence, AUC against the ground-truth masks, then the mean over the 15 categories. Note this is "pool within a category, then average", not "average per image".
- **PRO** (an addition in this project): per-connected-component recall, integrated and normalised to FPR ≤ 0.3. Pixel-level AUC is dominated by large defects — missing one small scratch in an image barely moves it. PRO weights every defect region equally, so missing a small defect costs as much as missing a large one, which is closer to what a production line cares about.

  PRO is on by default. **I got this wrong once**: a single-category smoke run appeared much slower with PRO enabled, so I wrote "roughly doubles the runtime" and set the default to off. Re-checking against the per-stage timings inside a run showed that slowdown came from machine contention (other experiments were running in parallel at the time), not from PRO:

  | | Wall clock | Feature extraction | Localisation | Rest (metrics + PRO + figures) |
  | --- | ---: | ---: | ---: | ---: |
  | PRO on | 19.5 min | 17.2 | 1.5 | **0.8** |
  | PRO off | 18.3 min | 15.7 | 2.1 | **0.5** |

  PRO costs about **18 seconds** across all 15 categories — around 2 % of a run (`per_region_overlap` breaks early once FPR exceeds the cap, so it never really walks all 100 thresholds). With the only reason for disabling it gone, the default was flipped to on.

  ```bash
  python -m spade.evaluate --categories bottle --compute-pro false    # explicitly off
  ```

  The reason the flag accepts both a bare form and a value is that `dvc.yaml` has no conditionals: it must emit `--compute-pro ${evaluate.compute_pro}` unconditionally, so the parameter has to be able to take a value.

---

## 6. Expected variation

- Run-to-run differences of up to a few hundredths of a point are normal across environments, arising from the PyTorch/torchvision version, the BLAS backend, and `cdist`'s matmul expansion path.
- The method is **free of randomness** (no training, no sampling, DataLoader does not shuffle), so repeated runs in the same environment are identical. Measured: running the full pipeline again returned both metrics bit-for-bit across all 15 categories, five times over. `config.seed` is hygiene only — nothing in the current implementation consumes it.
- `grid` reaches only 47.3 % image-level ROC-AUC, below chance. This is not a bug but an inherent limit of the approach on a regular repeating texture: `grid`'s global average-pooled descriptor simply does not separate normal from anomalous, because a defect changes almost nothing about the image-wide average.

  What is worth noticing is that **localisation does not collapse with it**: `grid` scores 98.35 % pixel-level ROC-AUC and 86.39 % PRO, both solidly above average. The two stages are decoupled — the global descriptor of stage 1 cannot pick out "which image has a problem", but as long as the K neighbours it selects are reasonable normal samples, the per-position correspondence of stage 2 still points at "where the problem is". On a line where the image-level verdict is unreliable, you can switch to the anomalous-area fraction instead; that is what `anomalous_pixel_ratio` in the `/predict` response is for.

- The two lowest PRO categories are `tile` (69.14 %) and `transistor` (70.76 %), consistent with their lower pixel-level ROC-AUC (92.88 % / 86.75 %). `transistor`'s defects are often a whole component being misplaced: the anomaly is in the global layout rather than local texture, and per-position nearest-neighbour search is inherently insensitive to that.

---

## 7. How K affects results

`K` — how many normal neighbours a test image is scored against — is the one hyper-parameter with real leverage. Running all 15 categories at `K=50` (`artifacts/runs/full-mvtec-k50/`) alongside the default `K=5` gave a result that was not what I expected:

| | Image ROC-AUC | Pixel ROC-AUC | PRO | Wall clock |
| --- | ---: | ---: | ---: | ---: |
| K=5 (default) | **85.41** | 96.44 | **86.13** | 19.4 min |
| K=50 | 80.88 | **96.98** | 83.87 | 91.1 min |

Two separate findings:

- **Pixel level: K=50 really is better — 96.98.** Localisation benefits from the larger gallery, as intuition suggests.
- **Image level: K=50 loses 4.53 points (85.41 → 80.88).** The two metrics move in opposite directions, so K is a genuine trade-off rather than a knob to turn up.

### Why the image level degrades

My first explanation was "K=50 covers too large a fraction of a small training set" (toothbrush has only 60 training images, so K=50 is 83 % of them). **The data rejected it**: the Pearson correlation between K/n_train and the image-level drop is only −0.19, and in the wrong direction — categories with *smaller* coverage dropped more (−6.70 vs −3.44), and the worst drop, `screw`, has only 15.6 % coverage.

The hypothesis that does hold is "categories whose image level was already weak at K=5 drop most", with correlation **+0.785**:

| Image ROC-AUC at K=5 | Categories | Mean drop at K=50 |
| --- | ---: | ---: |
| < 90 | 8 | **−7.57** |
| ≥ 90 | 7 | −1.05 |

The largest drops — `screw` (−15.96), `grid` (−11.86), `toothbrush` (−8.89), `metal_nut` (−8.65) — are exactly the categories with the worst image-level scores at K=5.

The mechanism is coherent: the image-level score is the mean distance to K nearest neighbours. Where the global descriptor already separates normal from anomalous (bottle 97.2, tile 96.5, zipper 96.6), averaging over a few more neighbours changes little; where it barely separates them at all (grid 47.3, screw 66.7), averaging 50 neighbours' distances washes out the little signal there was. **K=50 introduces no new failure — it amplifies the existing one.**

There is also a texture/object split (texture −2.94, object −5.32), but that is most likely a by-product of the above: four of the five texture categories already scored ≥ 92 at image level with K=5.

### Practical implication

If only **localisation** matters, larger K is better, at a cost that rises roughly linearly with K (a 10× larger gallery took the full run from 19.4 to 91.1 minutes, 4.7×). If an **image-level verdict** is needed, K=5 is the steadier setting. This project defaults to `K=5`, which serves both.

---

## 8. Environment

Every run records its environment in the `environment` field of `results.json` (torch version, platform). The numbers published in the README were obtained on:

- Apple M2 / 16 GB, macOS
- PyTorch 2.13.0 + torchvision 0.28.0, MPS backend
- Python 3.13

Changing device affects speed only, not results: re-running `bottle` and `grid` on CPU returned all four metrics (97.22 / 97.01 / 47.28 / 98.35) bit-for-bit against MPS.

One exception to "completely identical": the file hash of `artifacts/banks/*.pt` changes on every build, because the metadata carries a `created_utc` provenance timestamp. The feature tensors themselves are bit-for-bit identical — two independently built banks were compared in full with `torch.equal`, and `created_utc` was the only key that differed.

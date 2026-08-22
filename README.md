# Industrial Defect Detection & MLOps Pipeline

[![CI](https://github.com/miachillgood/industrial-defect-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/miachillgood/industrial-defect-detection-mlops/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

I built a **surface-defect detection system trained on defect-free samples only**: give it a photo of a part and it answers "is this defective?" and "which pixels exactly?". Around it I put a full production-shaped pipeline — DVC for data/model versioning, MLflow for experiment tracking, a FastAPI inference service, a Streamlit human-review loop, Docker, and GitHub Actions CI.

Across all 15 MVTec AD categories I get **85.41 % image-level ROC-AUC** and **96.44 % pixel-level ROC-AUC**.

![defect localisation example](artifacts/runs/full-mvtec-k5/images/hazelnut_001.png)

<sub>Left to right: input · ground-truth mask · anomaly heatmap · predicted mask · localised defect. The model never saw a single defective sample during setup.</sub>

---

## Why unsupervised

Defective samples are scarce on a production line, and the defect taxonomy is open-ended — the scratch, chip, or misalignment you have never seen will show up tomorrow anyway. A supervised detector would keep missing new defect types under that distribution, and every new product line would mean collecting and labelling a fresh batch of bad parts.

So I let the system learn only what **normal** looks like: a frozen ImageNet backbone encodes defect-free samples into a multi-scale feature bank (memory bank), a test image gets an anomaly score by nearest-neighbour retrieval, and a per-pixel correspondence over the feature pyramid localises the defect.

Three engineering consequences follow directly from that choice:

- **No weights to train.** Switching product lines means rebuilding the feature bank from a new set of normal samples — minutes, no hyper-parameter tuning, no waiting for convergence.
- **The decision threshold is a product decision, not a training outcome.** How to trade false negatives against false positives belongs to the business; the review tool I built exists to tune it.
- **Decisions are explainable.** Every verdict ships with a heatmap and a defect mask, so an inspector can see where the model is looking.

---

## What I built

| | |
| --- | --- |
| **Detection** | Wide-ResNet50-2 feature pyramid + kNN retrieval + deep pyramid correspondence, image-level and pixel-level output |
| **Evaluation** | Image / pixel ROC-AUC, PRO, ROC curves, per-category localisation figures |
| **Data versioning** | DVC tracks the 5 GB dataset and the multi-hundred-MB feature bank; git holds only hash pointers |
| **Experiment tracking** | MLflow records params, per-category metrics, and all artifacts — and **a tracking failure never breaks a run** |
| **Inference service** | FastAPI, lazy per-category bank loading, returns score + base64 heatmap + mask |
| **Human-in-the-loop** | Streamlit review console, append-only JSONL decisions, live view of how the threshold changes verdicts |
| **Delivery** | Multi-stage Dockerfile (CPU-only torch, non-root, baked-in weights) + docker-compose |
| **CI** | ruff, a Python 3.11/3.12 test matrix, API image smoke test |
| **Tests** | 141 tests, 97 % line coverage over `src/`, all runnable without the 5 GB dataset |

The dataset is [MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad): 15 categories of industrial products and materials, 3 629 training images + 1 725 test images.

---

## Method

Two stages, and **no parameter is ever updated**.

### Stage 1 · Image-level retrieval

Each image is summarised by the 2048-d global average-pooled Wide-ResNet50-2 descriptor. A test image's anomaly score is the mean Euclidean distance to its **K nearest normal samples**:

```
score(x) = mean( topK_min ‖ f_avg(x) − f_avg(t_i) ‖₂ ,  t_i ∈ normal bank )
```

### Stage 2 · Pixel-level deep pyramid correspondence

Reusing the same K neighbours, I flatten their feature vectors at **every spatial position** across `layer1 / layer2 / layer3` into one gallery. Each test position scores as its distance to the closest gallery vector; the three scales are bilinearly upsampled to 224×224, averaged, and smoothed with a σ=4 Gaussian.

Resolutions: `layer1` 56×56 (256 channels), `layer2` 28×28 (512), `layer3` 14×14 (1024). At K=5 the `layer1` gallery is 5×56×56 = 15 680 vectors of dimension 256.

I default to `K=5`; a larger K means a larger gallery and steadier localisation, at proportionally higher cost.

---

## Results

<!-- RESULTS:BEGIN -->
<!-- source: artifacts/runs/full-mvtec-k5/results.json -->
All 15 MVTec AD categories, `K=5`, `wide_resnet50_2` (ImageNet **IMAGENET1K_V1** weights, frozen throughout):

| | Image ROC-AUC | Pixel ROC-AUC | PRO |
| --- | ---: | ---: | ---: |
| **Mean over 15 categories** | **85.41 %** | **96.44 %** | **86.13 %** |

PRO (per-region overlap, integrated to FPR <= 0.3) weights every defect region equally, unlike pixel ROC-AUC which large defects dominate: missing a small scratch costs as much as missing a large one.

<details>
<summary>Per-category detail (click to expand)</summary>

| category | image ROC-AUC | pixel ROC-AUC | PRO |
| --- | ---: | ---: | ---: |
| bottle | 97.22 | 97.01 | 93.33 |
| cable | 84.84 | 92.41 | 73.51 |
| capsule | 89.67 | 98.39 | 90.21 |
| carpet | 92.78 | 98.91 | 80.83 |
| grid | 47.28 | 98.35 | 86.39 |
| hazelnut | 88.14 | 98.53 | 93.39 |
| leather | 95.38 | 99.32 | 97.37 |
| metal_nut | 70.97 | 97.14 | 86.76 |
| pill | 80.14 | 94.98 | 91.78 |
| screw | 66.71 | 99.11 | 93.45 |
| tile | 96.50 | 92.88 | 69.14 |
| toothbrush | 88.89 | 98.85 | 91.74 |
| transistor | 90.25 | 86.75 | 70.76 |
| wood | 95.79 | 95.33 | 91.66 |
| zipper | 96.59 | 98.59 | 81.67 |
| **Mean** | **85.41** | **96.44** | **86.13** |

</details>

> Environment: macOS-26.5.2-arm64-arm-64bit-Mach-O, PyTorch 2.13.0, device `mps`, 19.4 min end to end.
> The method neither trains nor samples, so repeated runs in the same environment are bit-for-bit identical.

`grid` scores 47 % at image level -- below chance. That is a known limit of global-descriptor retrieval on a regular texture, not a bug: localisation on the same category is fine (98.35 % pixel, 86.39 % PRO). See [docs/method.md](docs/method.md#6-expected-variation).
<!-- RESULTS:END -->

![ROC curves](artifacts/runs/full-mvtec-k5/roc_curve.png)

### How K matters

`K` — how many normal neighbours a test image is scored against — is the one hyper-parameter with real leverage, so I ran the full benchmark at both ends (`artifacts/runs/full-mvtec-k50/`). It pulls the two metrics in opposite directions:

| | Image ROC-AUC | Pixel ROC-AUC | Wall clock |
| --- | ---: | ---: | ---: |
| K=5 | **85.41** | 96.44 | 19.4 min |
| K=50 | 80.88 | **96.98** | 91.1 min |

- **Localisation improves with K** — 96.98 at K=50, from a gallery 10× larger.
- **Image-level detection degrades by 4.53 points**, and costs 4.7× the wall clock to do it.

I got the explanation wrong on the first try. My guess was "K=50 covers too large a fraction of a small training set", but K/n_train correlates with the drop at only −0.19, and in the wrong direction. The hypothesis that holds is "categories whose image-level score was already weak at K=5 degrade most", with correlation **+0.785** — the 8 categories below 90 lose 7.57 points on average, the 7 at or above 90 only 1.05. A larger K introduces no new failure; it averages what little signal there was across 50 neighbours.

So `K=5` is the default here: it is the setting that serves both metrics at once. Details in [docs/method.md](docs/method.md#7-how-k-affects-results).

---

## Quick start

```bash
make venv && make install
```

### 1. Prepare the data (~5 GB)

MVTec AD is released under CC BY-NC-SA 4.0 for non-commercial research only. Please read and accept the licence on the [official page](https://www.mvtec.com/company/research/datasets/mvtec-ad) first.

```bash
make data
```

`scripts/prepare_data.py` accepts a mirror URL or a local archive, then verifies the per-category image counts after extraction and exits non-zero if anything is missing:

```bash
python scripts/prepare_data.py --archive ~/Downloads/mvtec_anomaly_detection.tar.xz
python scripts/prepare_data.py --check-only   # verify layout and counts only
```

### 2. Run all 15 categories

```bash
make eval
make eval-quick     # single-category smoke run (bottle)
```

Outputs land in `artifacts/runs/<run-name>/`: `results.md` (comparison tables), `results.json`, `metrics.json`, `roc_curve.png`, and `images/` (localisation figures).

### 3. Build a deployable memory bank

```bash
make bank           # bottle by default -> artifacts/banks/spade_bottle.pt
```

Thresholds are calibrated **inside the training split** by leave-one-out: score each normal image against its K nearest neighbours excluding itself, take the resulting normal-score distribution, and pick a percentile (image-level P99, pixel-level P99.5 by default). **The test split is never touched**, so the ROC-AUC figures above are not contaminated by threshold calibration.

### 4. Run the services

```bash
make api            # FastAPI   -> http://localhost:8000/docs
make review         # Streamlit -> http://localhost:8501
make mlflow         # MLflow UI -> http://localhost:5000
```

All at once:

```bash
docker compose -f docker/docker-compose.yml up --build
```

> If the repository path contains non-ASCII characters, `docker compose build` fails with a gRPC "non-printable ASCII char" error — a Compose/buildx limitation, not a Dockerfile problem. Build with `docker build` and start with `--no-build`; see [docs/mlops.md](docs/mlops.md#known-limitation-a-non-ascii-path-breaks-docker-compose-build).

Calling the inference API:

```bash
curl -s -F 'file=@data/mvtec_anomaly_detection/bottle/test/broken_large/000.png' \
  'http://localhost:8000/predict?category=bottle&include_images=false' | jq
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness probe + loaded categories |
| `GET /models` | Per-bank metadata and calibration details |
| `GET /stats` | Request count, anomaly count, mean latency |
| `POST /predict` | Score, verdict, base64 heatmap and mask |
| `POST /predict/overlay` | The blended overlay straight back as a PNG |

---

## Architecture

```
prepare_data ──► evaluate ──► build_bank ──► FastAPI service
     │              │              │              │
    DVC           MLflow          DVC        Streamlit review
 (data version) (experiments)  (model version)  (human feedback)
                                                  │
                                 threshold recalibration ◄───┘
```

The premise behind this pipeline: **the method has no checkpoint** — the "model" is a frozen backbone plus a bank of normal-sample features. What needs versioning is therefore **data**, not weights, which is exactly why I reached for DVC rather than a model registry. Change `evaluate.top_k` in `params.yaml` and `dvc repro` re-runs only `evaluate` and `build_bank`, not the data verification.

I made the MLflow layer **fail-soft on purpose**: whether MLflow is installed or its server is reachable, `make eval` must still finish and write `results.json`. Tracking is observation, not a runtime dependency.

---

## Layout

```
src/spade/            detection core
  config.py           hyper-parameters and device selection
  data.py             MVTec AD dataset and pre-processing
  features.py         Wide-ResNet50-2 feature pyramid (forward hooks)
  model.py            kNN retrieval + deep pyramid correspondence
  metrics.py          ROC-AUC / PRO / reference tables
  evaluate.py         per-category evaluation and report generation
  inference.py        single-image inference (shared by API and review tool)
  cache.py            on-disk feature cache
  visualize.py        ROC curves and localisation figures
src/mlops/            engineering layer
  tracking.py         MLflow wrapper (fail-soft)
  review_store.py     human review decisions (append-only JSONL)
apps/api/             FastAPI inference service
apps/streamlit/       annotation review tool
scripts/              data preparation, bank building, README result sync
docker/               Dockerfile + docker-compose
tests/                unit tests and documentation attribution guards
docs/                 method details, engineering notes
```

---

## Four traps I hit and handled explicitly

The full list is in [docs/method.md](docs/method.md); these four matter most:

1. **`torch.pairwise_distance` changed its reduction axis in PyTorch 2.x.**
   On PyTorch 1.x it reduced over `dim=1` (channels); from 2.0 it reduces over the **last** axis — the same line **silently** computes a completely different quantity on a modern install, with no error. I reduce over the channel axis explicitly and pin the semantics with `tests/test_model.py::test_channelwise_reduction_is_the_intended_distance`, so nobody "simplifies" it back to the built-in.

2. **The ImageNet weights must be `IMAGENET1K_V1`.**
   The V2 weights torchvision added later were retrained with an improved recipe; the feature distribution differs and every number moves. I hard-code V1 in `features.py` and raise on an unsupported backbone rather than silently degrading.

3. **`Image.ANTIALIAS` is LANCZOS, not bilinear.**
   The `Resize(256)` in pre-processing must use LANCZOS resampling; switching to the default bilinear shifts results systematically.

4. **Chunking the gallery with integer division silently drops its tail.**
   The obvious loop, `range(gallery_size // 100)`, discards the last `G % 100` rows — at K=5 that is 80 of layer1's 15 680, quietly inflating every nearest-neighbour distance. I use the full gallery, but I did not stop at "the effect is probably small": I ran all 15 categories both ways. Truncation shifts the pixel-level mean by 0.04 and moves `transistor` by 0.17, while image level is untouched (its score comes from avgpool and never passes through the gallery). `--drop-gallery-remainder` keeps the truncated mode available so the difference stays measurable rather than assumed; both runs are committed under `artifacts/runs/`.

---

## Documentation

- [docs/method.md](docs/method.md) — method and implementation detail: the two stages, the full configuration table, the four traps with measured impact, metric definitions, expected variation, the K ablation, environment
- [docs/mlops.md](docs/mlops.md) — engineering layer: the DVC pipeline, MLflow tracking, FastAPI/Streamlit design trade-offs, Docker and CI

---

## Credits

All code here is my own. The two-stage retrieval approach it implements is SPADE, from Niv Cohen and Yedid Hoshen, [Sub-Image Anomaly Detection with Deep Pyramid Correspondences](https://arxiv.org/abs/2005.02357) (arXiv:2005.02357); [`byungjae89/SPADE-pytorch`](https://github.com/byungjae89/SPADE-pytorch) (MIT) was a useful cross-check while settling configuration details.

> Not to be confused with [NVlabs/SPADE](https://github.com/NVlabs/SPADE), a **completely unrelated** method — spatially-adaptive normalization for image synthesis (GauGAN). The acronym collision is the only connection.

```bibtex
@article{cohen2020sub,
  title   = {Sub-Image Anomaly Detection with Deep Pyramid Correspondences},
  author  = {Cohen, Niv and Hoshen, Yedid},
  journal = {arXiv preprint arXiv:2005.02357},
  year    = {2020}
}

@inproceedings{bergmann2019mvtec,
  title     = {MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author    = {Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle = {CVPR},
  year      = {2019}
}
```

## Licence

The code in this repository is released under the MIT licence, see [LICENSE](LICENSE).
The MVTec AD dataset is not distributed here; it is licensed CC BY-NC-SA 4.0 (non-commercial research only). The sample images shown in this README come from that dataset and are used for non-commercial illustration of the method.

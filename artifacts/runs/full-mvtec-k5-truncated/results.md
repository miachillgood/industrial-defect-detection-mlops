# Defect detection results

- backbone: `wide_resnet50_2` (ImageNet **IMAGENET1K_V1** weights, frozen)
- K (nearest neighbours): **5**
- input: resize 256 (LANCZOS) -> center-crop 224
- device: `mps`

Reference = `byungjae89/SPADE-pytorch` README (K=5), used as a public baseline. `delta` is ours minus reference, in ROC-AUC points.

## Image-level ROC-AUC (%)

| category | reference | ours | delta |
| --- | ---: | ---: | ---: |
| bottle | 97.2 | 97.22 | +0.02 |
| cable | 84.8 | 84.84 | +0.04 |
| capsule | 89.7 | 89.67 | -0.03 |
| carpet | 92.8 | 92.78 | -0.02 |
| grid | 47.3 | 47.28 | -0.02 |
| hazelnut | 88.1 | 88.14 | +0.04 |
| leather | 95.4 | 95.38 | -0.02 |
| metal_nut | 71.0 | 70.97 | -0.03 |
| pill | 80.1 | 80.14 | +0.04 |
| screw | 66.7 | 66.71 | +0.01 |
| tile | 96.5 | 96.50 | +0.00 |
| toothbrush | 88.9 | 88.89 | -0.01 |
| transistor | 90.3 | 90.25 | -0.05 |
| wood | 95.8 | 95.79 | -0.01 |
| zipper | 96.6 | 96.59 | -0.01 |
| **Average** | **85.4** | **85.41** | **+0.01** |

## Pixel-level ROC-AUC (%)

| category | reference | ours | delta |
| --- | ---: | ---: | ---: |
| bottle | 97.0 | 96.99 | -0.01 |
| cable | 92.3 | 92.32 | +0.02 |
| capsule | 98.4 | 98.38 | -0.02 |
| carpet | 98.9 | 98.91 | +0.01 |
| grid | 98.3 | 98.32 | +0.02 |
| hazelnut | 98.5 | 98.52 | +0.02 |
| leather | 99.3 | 99.32 | +0.02 |
| metal_nut | 97.1 | 97.06 | -0.04 |
| pill | 95.0 | 94.99 | -0.01 |
| screw | 99.1 | 99.10 | +0.00 |
| tile | 92.8 | 92.83 | +0.03 |
| toothbrush | 98.8 | 98.83 | +0.03 |
| transistor | 86.6 | 86.58 | -0.02 |
| wood | 95.3 | 95.31 | +0.01 |
| zipper | 98.6 | 98.55 | -0.05 |
| **Average** | **96.4** | **96.40** | **+0.00** |

# Defect detection results

- backbone: `wide_resnet50_2` (ImageNet **IMAGENET1K_V1** weights, frozen)
- K (nearest neighbours): **50**
- input: resize 256 (LANCZOS) -> center-crop 224
- device: `mps`

Reference = `byungjae89/SPADE-pytorch` README (K=5), used as a public baseline. `delta` is ours minus reference, in ROC-AUC points.

## Image-level ROC-AUC (%)

| category | reference | ours | delta |
| --- | ---: | ---: | ---: |
| bottle | 97.2 | 96.35 | -0.85 |
| cable | 84.8 | 81.99 | -2.81 |
| capsule | 89.7 | 82.61 | -7.09 |
| carpet | 92.8 | 92.42 | -0.38 |
| grid | 47.3 | 35.42 | -11.88 |
| hazelnut | 88.1 | 84.04 | -4.06 |
| leather | 95.4 | 93.31 | -2.09 |
| metal_nut | 71.0 | 62.32 | -8.68 |
| pill | 80.1 | 78.94 | -1.16 |
| screw | 66.7 | 50.75 | -15.95 |
| tile | 96.5 | 95.74 | -0.76 |
| toothbrush | 88.9 | 80.00 | -8.90 |
| transistor | 90.3 | 87.46 | -2.84 |
| wood | 95.8 | 96.14 | +0.34 |
| zipper | 96.6 | 95.77 | -0.83 |
| **Average** | **85.4** | **80.88** | **-4.52** |

## Pixel-level ROC-AUC (%)

| category | reference | ours | delta |
| --- | ---: | ---: | ---: |
| bottle | 97.0 | 97.63 | +0.63 |
| cable | 92.3 | 93.71 | +1.41 |
| capsule | 98.4 | 98.67 | +0.27 |
| carpet | 98.9 | 99.07 | +0.17 |
| grid | 98.3 | 98.73 | +0.43 |
| hazelnut | 98.5 | 98.73 | +0.23 |
| leather | 99.3 | 99.34 | +0.04 |
| metal_nut | 97.1 | 97.35 | +0.25 |
| pill | 95.0 | 95.65 | +0.65 |
| screw | 99.1 | 99.31 | +0.21 |
| tile | 92.8 | 93.95 | +1.15 |
| toothbrush | 98.8 | 98.87 | +0.07 |
| transistor | 86.6 | 89.15 | +2.55 |
| wood | 95.3 | 95.69 | +0.39 |
| zipper | 98.6 | 98.92 | +0.32 |
| **Average** | **96.4** | **96.98** | **+0.58** |

## PRO (%)

Per-region overlap, integrated to FPR <= 0.3. Every ground-truth defect region counts equally, so a missed small defect costs as much as a missed large one -- unlike pixel ROC-AUC, which large defects dominate. The public baseline does not report PRO, so there is no reference column.

| category | PRO |
| --- | ---: |
| bottle | 86.41 |
| cable | 78.05 |
| capsule | 84.10 |
| carpet | 76.71 |
| grid | 87.59 |
| hazelnut | 90.51 |
| leather | 89.61 |
| metal_nut | 83.27 |
| pill | 88.76 |
| screw | 96.45 |
| tile | 66.53 |
| toothbrush | 91.75 |
| transistor | 78.29 |
| wood | 78.43 |
| zipper | 81.63 |
| **Average** | **83.87** |

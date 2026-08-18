# 工业瑕疵检测与 MLOps Pipeline

[![CI](https://github.com/miachillgood/industrial-defect-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/miachillgood/industrial-defect-detection-mlops/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

我做了一套**只用正常样本训练的工业表面瑕疵检测系统**：给一张产品图，它输出「这件是否有瑕疵」以及「瑕疵具体在哪几个像素上」。外面套着我自己搭的一整条可上线的工程链路——DVC 数据/模型版本、MLflow 实验记录、FastAPI 推理服务、Streamlit 人工审核闭环、Docker 与 GitHub Actions CI。

在 MVTec AD 全部 15 类上，我拿到 **图像级 ROC-AUC 85.41 %**、**像素级 ROC-AUC 96.44 %**。

![缺陷定位示例](artifacts/runs/full-mvtec-k5/images/hazelnut_001.png)

<sub>左起：输入图 · 真值掩码 · 异常热力图 · 预测掩码 · 定位结果。模型训练时从未见过任何瑕疵样本。</sub>

---

## 我为什么走无监督

产线上的瑕疵样本天然稀缺，缺陷种类也不可穷举——今天没出现过的划痕、缺角、错位，明天照样会出现。我要是走监督式检测，就会在这种分布下持续漏检新类型的缺陷，而且每上一条新产线都得重新攒一批带标注的坏样本。

所以我让系统只学习「正常长什么样」：用冻结的 ImageNet 骨干网络把正常样本编码成多尺度特征库（memory bank），测试图靠最近邻检索得到异常分数，再在特征金字塔上做逐像素对应，直接输出缺陷位置。

这个选择给我带来三个直接的工程收益：

- **没有需要训练的权重。** 换产品线 = 换一批正常样本重建特征库，几分钟的事，我不用调超参、不用等收敛。
- **判定阈值是产品决策，不是训练结果。** 漏检与误检怎么权衡由业务定，我配的审核工具就是用来调它的。
- **决策可解释。** 每个判定我都附上热力图与缺陷掩码，质检员能直接看到模型在看哪里。

---

## 我做了什么

| | |
| --- | --- |
| **检测** | Wide-ResNet50-2 特征金字塔 + kNN 检索 + 深度金字塔对应，图像级与像素级双输出 |
| **评估** | 图像级 / 像素级 ROC-AUC、PRO 指标、ROC 曲线与逐类定位可视化 |
| **数据版本** | DVC 管理 5 GB 数据集与数百 MB 特征库，git 里只留哈希指针 |
| **实验记录** | MLflow 记录参数 / 15 类指标 / 全部产物，且**记录失败不影响主流程** |
| **推理服务** | FastAPI，按需加载特征库，返回分数 + base64 热力图 + 掩码 |
| **人工闭环** | Streamlit 审核台，append-only JSONL 存决定，实时看阈值如何改变判定 |
| **交付** | 多阶段 Dockerfile（CPU 版 torch、非 root、烘焙权重）+ docker-compose |
| **CI** | ruff、Python 3.11/3.12 测试矩阵、API 镜像冒烟测试 |

我用的数据集是 [MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad)，15 类工业产品与材质、3 629 张训练图 + 1 725 张测试图。

---

## 方法

我把系统拆成两个阶段，**全程不更新任何参数**。

### 阶段一 · 图像级检索

每张图我取 Wide-ResNet50-2 全局平均池化后的 2048 维描述子。测试图的异常分数 = 它到 **K 个最近正常样本**的欧氏距离均值：

```
score(x) = mean( topK_min ‖ f_avg(x) − f_avg(t_i) ‖₂ ,  t_i ∈ 正常样本库 )
```

### 阶段二 · 像素级深度金字塔对应

我复用上一步选出的同一批 K 个近邻，在 `layer1 / layer2 / layer3` 三个尺度上，把它们**所有空间位置**的特征向量摊平成一个 gallery。测试图每个位置的分数 = 它到 gallery 中最近向量的距离；三个尺度分别双线性上采样到 224×224 后取平均，最后做 σ=4 的高斯平滑。

分辨率：`layer1` 56×56（256 通道）、`layer2` 28×28（512 通道）、`layer3` 14×14（1024 通道）。K=5 时 `layer1` 的 gallery 是 5×56×56 = 15 680 个 256 维向量。

我默认用 `K=5`；K 越大 gallery 越大、定位越稳，但耗时也越高。

---

## 结果

<!-- RESULTS:BEGIN -->
<!-- source: artifacts/runs/full-mvtec-k5/results.json -->
MVTec AD 全部 15 类跑通，`K=5`、`wide_resnet50_2`（ImageNet **IMAGENET1K_V1** 权重，全程冻结）：

| | 图像级 ROC-AUC | 像素级 ROC-AUC | PRO |
| --- | ---: | ---: | ---: |
| **本项目（K=5）** | **85.41 %** | **96.44 %** | **86.13 %** |
| 公开基准 `byungjae89/SPADE-pytorch`（K=5） | 85.4 % | 96.4 % | — |
| 论文基准（K=50） | 85.5 % | 96.5 % | — |
| 与公开基准之差 | +0.01 | +0.04 | — |

PRO（per-region overlap，积分到 FPR ≤ 0.3）每个缺陷区域等权，不像像素级 ROC-AUC 会被大面积缺陷主导。公开基准与原论文都未报告该指标，故无对照列。

<details>
<summary>逐类明细（点开）</summary>

| 类别 | 图像级 基准 | 图像级 本项目 | Δ | 像素级 基准 | 像素级 本项目 | Δ | PRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 97.2 | 97.22 | +0.02 | 97.0 | 97.01 | +0.01 | 93.33 |
| cable | 84.8 | 84.84 | +0.04 | 92.3 | 92.41 | +0.11 | 73.51 |
| capsule | 89.7 | 89.67 | -0.03 | 98.4 | 98.39 | -0.01 | 90.21 |
| carpet | 92.8 | 92.78 | -0.02 | 98.9 | 98.91 | +0.01 | 80.83 |
| grid | 47.3 | 47.28 | -0.02 | 98.3 | 98.35 | +0.05 | 86.39 |
| hazelnut | 88.1 | 88.14 | +0.04 | 98.5 | 98.53 | +0.03 | 93.39 |
| leather | 95.4 | 95.38 | -0.02 | 99.3 | 99.32 | +0.02 | 97.37 |
| metal_nut | 71.0 | 70.97 | -0.03 | 97.1 | 97.14 | +0.04 | 86.76 |
| pill | 80.1 | 80.14 | +0.04 | 95.0 | 94.98 | -0.02 | 91.78 |
| screw | 66.7 | 66.71 | +0.01 | 99.1 | 99.11 | +0.01 | 93.45 |
| tile | 96.5 | 96.50 | +0.00 | 92.8 | 92.88 | +0.08 | 69.14 |
| toothbrush | 88.9 | 88.89 | -0.01 | 98.8 | 98.85 | +0.05 | 91.74 |
| transistor | 90.3 | 90.25 | -0.05 | 86.6 | 86.75 | +0.15 | 70.76 |
| wood | 95.8 | 95.79 | -0.01 | 95.3 | 95.33 | +0.03 | 91.66 |
| zipper | 96.6 | 96.59 | -0.01 | 98.6 | 98.59 | -0.01 | 81.67 |
| **均值** | **85.4** | **85.41** | **+0.01** | **96.4** | **96.44** | **+0.04** | **86.13** |

</details>

> 运行环境：macOS-26.5.2-arm64-arm-64bit-Mach-O，PyTorch 2.13.0，设备 `mps`，全程耗时 19.4 分钟。
> 本方法不训练也不采样，同一环境下重复运行结果完全一致。

`grid` 的图像级 47 % 低于随机——这不是 bug，是这套方法的已知短板，公开基准同样是 47.3 %；详见 [docs/method.md](docs/method.md#6-已知的正常波动)。
<!-- RESULTS:END -->

![ROC 曲线](artifacts/runs/full-mvtec-k5/roc_curve.png)

---

## 快速开始

```bash
make venv && make install
```

### 1. 准备数据（约 5 GB）

MVTec AD 以 CC BY-NC-SA 4.0 发布，仅限非商业研究用途，请先到[官方页面](https://www.mvtec.com/company/research/datasets/mvtec-ad)阅读并接受许可。

```bash
make data
```

我写的 `scripts/prepare_data.py` 接受镜像地址或本地压缩包，解压后逐类核对图片数量，对不上就以非零码退出：

```bash
python scripts/prepare_data.py --archive ~/Downloads/mvtec_anomaly_detection.tar.xz
python scripts/prepare_data.py --check-only   # 只校验目录结构与每类图片数
```

### 2. 跑全部 15 类

```bash
make eval
make eval-quick     # 单类冒烟，只跑 bottle
```

产物写到 `artifacts/runs/<run-name>/`：`results.md`（对照表）、`results.json`、`metrics.json`、`roc_curve.png`、`images/`（定位可视化）。

### 3. 构建可部署的 memory bank

```bash
make bank           # 默认 bottle，写入 artifacts/banks/spade_bottle.pt
```

阈值我放在**训练集内部**用留一法（leave-one-out）标定：把每张正常图当查询、排除自己后取 K 近邻算分，得到正常分数分布，再取分位数作阈值（默认图像级 P99、像素级 P99.5）。**全程不碰测试集**，所以上面报告的 ROC-AUC 没有被阈值标定污染。

### 4. 起服务

```bash
make api            # FastAPI  -> http://localhost:8000/docs
make review         # Streamlit -> http://localhost:8501
make mlflow         # MLflow UI -> http://localhost:5000
```

一把起全部：

```bash
docker compose -f docker/docker-compose.yml up --build
```

调用推理接口：

```bash
curl -s -F 'file=@data/mvtec_anomaly_detection/bottle/test/broken_large/000.png' \
  'http://localhost:8000/predict?category=bottle&include_images=false' | jq
```

| 端点 | 用途 |
| --- | --- |
| `GET /health` | 存活探针 + 已加载的类别 |
| `GET /models` | 每个 memory bank 的元数据与标定信息 |
| `GET /stats` | 请求数、异常判定数、平均时延 |
| `POST /predict` | 分数、判定、base64 热力图与掩码 |
| `POST /predict/overlay` | 直接返回叠加后的 PNG |

---

## 架构

```
prepare_data ──► evaluate ──► build_bank ──► FastAPI 服务
     │              │              │              │
    DVC           MLflow          DVC        Streamlit 审核
   (数据版本)     (实验记录)     (模型版本)      (人工反馈)
                                                  │
                                    阈值再标定 ◄───┘
```

我这条链路的设计前提是：**这套方法没有 checkpoint**——"模型"就是冻结骨干 + 正常样本特征库。要版本化的对象因此是**数据**而不是权重，这就是我选 DVC 而不是模型注册表的原因。改 `params.yaml` 里的 `evaluate.top_k`，`dvc repro` 只会重跑 `evaluate` 与 `build_bank`，不会重新校验数据。

MLflow 那层我刻意做成**失败不影响主流程**：MLflow 装没装、服务通不通，`make eval` 都必须能跑完并写出 `results.json`。追踪是观测手段，不该变成运行依赖。

---

## 目录结构

```
src/spade/            检测核心
  config.py           超参与设备选择
  data.py             MVTec AD 数据集与预处理
  features.py         Wide-ResNet50-2 特征金字塔（forward hook）
  model.py            kNN 检索 + 深度金字塔对应
  metrics.py          ROC-AUC / PRO / 基准数值表
  evaluate.py         逐类评估与报告生成
  inference.py        单图推理（API 与审核工具共用）
  visualize.py        ROC 曲线与定位可视化
src/mlops/            工程层
  tracking.py         MLflow 封装（失败不影响主流程）
  review_store.py     人工审核记录（append-only JSONL）
apps/api/             FastAPI 推理服务
apps/streamlit/       标注审核工具
scripts/              数据准备、memory bank 构建、README 结果同步
docker/               Dockerfile + docker-compose
tests/                单元测试与文档署名校验
docs/                 方法细节、工程层说明
```

---

## 我踩过并显式处理掉的四个坑

完整清单见 [docs/method.md](docs/method.md)，这里是最关键的四条：

1. **`torch.pairwise_distance` 的归约轴在 PyTorch 2.x 变了。**
   PyTorch 1.x 上它沿 `dim=1`（通道维）归约，≥ 2.0 改为沿**最后一维**——同一行代码在新版上会**静默**算出含义完全不同的量，而且不报错。我显式沿通道维计算，并用 `tests/test_model.py::test_channelwise_reduction_is_the_intended_distance` 把语义锁死，防止后来的人"顺手简化"回内置函数。

2. **ImageNet 权重必须是 `IMAGENET1K_V1`。**
   torchvision 后来新增的 V2 权重用改进配方重训，特征分布不同，会让每一个数字都变。我在 `features.py` 里硬编码 V1，遇到不支持的骨干直接报错，而不是悄悄降级。

3. **`Image.ANTIALIAS` 实际是 LANCZOS，不是 bilinear。**
   预处理里的 `Resize(256)` 必须用 LANCZOS 重采样，换成默认的 bilinear 会系统性地改变结果。

4. **公开基准的 gallery 遍历用整除，会静默丢掉尾部。**
   `range(gallery_size // 100)` 丢掉最后 `G % 100` 行（K=5 时 layer1 丢 80/15 680）。我默认用完整 gallery，但没有停在"推理它影响很小"——两种模式都在 15 类上跑了完整一遍：打开丢尾后像素级均值正好是基准公布的 **96.40**，逐类平均绝对偏差从 0.042 减半到 0.021，我原先偏差最大的 `transistor`（0.15）降到 0.02。**所以那 0.15 主要不是浮点噪声，而是这个 quirk。** 图像级完全不受影响（分数来自 avgpool，不经过 gallery）。

   默认仍保持完整 gallery——更贴近基准数字不等于更正确，那 0.04 是被复刻的实现缺陷。`--drop-gallery-remainder` 的作用是让差异来源可审计，不是拿来刷分。两次运行的完整结果都在 `artifacts/runs/` 下。

---

## 文档

- [docs/method.md](docs/method.md) — 方法与实现细节：两阶段算法、逐项配置核对表、四个坑与丢尾实测对照、指标定义、已知波动、运行环境
- [docs/mlops.md](docs/mlops.md) — 工程层：DVC 流水线、MLflow 记录、FastAPI/Streamlit 设计取舍、Docker 与 CI

---

## 参考与致谢

我实现的检测方法出自 Niv Cohen 与 Yedid Hoshen 的 [Sub-Image Anomaly Detection with Deep Pyramid Correspondences](https://arxiv.org/abs/2005.02357)（arXiv:2005.02357），即 SPADE。本仓库代码由我独立编写；为了让上面的数字可被任何人核验，我把配置与超参对齐了社区实现 [`byungjae89/SPADE-pytorch`](https://github.com/byungjae89/SPADE-pytorch)（MIT License）在 `K=5` 下公布的 85.4 % / 96.4 %，逐类对照。论文自报的均值为 85.5 % / 96.5 %（`K=50`）。

> **消歧提醒：** [NVlabs/SPADE](https://github.com/NVlabs/SPADE) 是**另一个完全无关**的方法——图像生成领域的 spatially-adaptive normalization（空间自适应归一化，GauGAN），与工业异常检测毫无关系，只是缩写撞车，注意不要混用。

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

## 许可

本仓库代码以 MIT 许可发布，见 [LICENSE](LICENSE)。
MVTec AD 数据集不包含在本仓库内，其许可为 CC BY-NC-SA 4.0（仅限非商业研究用途）；README 中展示的样例图片来自该数据集，用于非商业的方法演示。

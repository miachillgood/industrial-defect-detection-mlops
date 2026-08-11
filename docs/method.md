# 方法与实现细节

本文件记录：检测算法怎么实现的、每一项配置为什么这么定、哪些地方与常见写法不同以及为什么。数值口径与 [byungjae89/SPADE-pytorch](https://github.com/byungjae89/SPADE-pytorch)（`K=5`）逐项核对，方便任何人复核 README 里的数字。

---

## 1. 算法：两阶段检索

这套方法**不训练任何参数**。所谓"模型"是一个冻结的 ImageNet 分类器，加上正常（无瑕疵）训练图片的特征库。

### 阶段一：图像级检索

对每张图取 Wide-ResNet50-2 全局平均池化后的 2048 维描述子。测试图的异常分数 = 它到 **K 个最近训练图**的欧氏距离均值。

```
score(x) = mean( topK_min ‖ f_avg(x) − f_avg(t_i) ‖₂ ,  t_i ∈ train )
```

### 阶段二：像素级深度金字塔对应

取上一步选出的同一批 K 个近邻，在 `layer1 / layer2 / layer3` 三个尺度上，把这 K 张图**所有空间位置**的特征向量摊平成一个 gallery。测试图每个位置的分数 = 它到 gallery 中最近向量的距离；三个尺度各自双线性上采样到 224×224 后取平均，最后做 σ=4 的高斯平滑。

分辨率：`layer1` 56×56（256 通道）、`layer2` 28×28（512 通道）、`layer3` 14×14（1024 通道）。K=5 时 `layer1` 的 gallery 是 5×56×56 = 15 680 个 256 维向量。

---

## 2. 逐项配置核对表

下表把每一项配置与公开基准 `byungjae89/SPADE-pytorch`（`K=5`）对照，用来解释 README 里 ±0.05 以内的差异是从哪来的。

| 环节 | 公开基准 | 本项目 | 一致 |
| --- | --- | --- | :---: |
| 骨干网络 | `wide_resnet50_2(pretrained=True)` | `weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1` | ✅ |
| Hook 位置 | `layer1[-1] / layer2[-1] / layer3[-1] / avgpool` | 同 | ✅ |
| 缩放 | `T.Resize(256, Image.ANTIALIAS)` | `InterpolationMode.LANCZOS` | ✅ |
| 裁剪 | `T.CenterCrop(224)` | 同 | ✅ |
| 归一化 | ImageNet mean/std | 同 | ✅ |
| 掩码插值 | `Image.NEAREST` | `InterpolationMode.NEAREST` | ✅ |
| K | 5 | 5（`--top-k` 可改） | ✅ |
| 图像级距离 | 展平 avgpool 后的欧氏距离 | `torch.cdist` | ✅ |
| 图像级分数 | top-K 距离取均值 | 同 | ✅ |
| 像素级尺度 | layer1/2/3 | 同 | ✅ |
| 融合方式 | 三尺度上采样后取均值 | 同 | ✅ |
| 平滑 | `gaussian_filter(sigma=4)` | 同 | ✅ |
| 距离 eps | `‖x₁ − x₂ + 1e-6‖` | 等价改写为 `‖(x₁+1e-6) − x₂‖` | ✅ |
| gallery 遍历 | `range(G // 100)`，丢弃尾部 | 使用完整 gallery（见 §3.3） | ⚠️ 有意偏离 |

---

## 3. 三个必须显式处理的坑

### 3.1 `torch.pairwise_distance` 在 PyTorch 2.x 换了归约轴

常见的写法是这一行：

```python
dist_matrix = torch.pairwise_distance(feat_gallery[a:b], test_feat_map)
# feat_gallery : (100, C, 1, 1)
# test_feat_map: (1,   C, H, W)
```

它依赖广播后**沿 `dim=1`（通道维）**求范数，得到 `(100, H, W)`——这在 PyTorch 1.x 是对的。PyTorch ≥ 2.0 的 `pairwise_distance` 改为沿**最后一维**归约，同样的输入会得到 `(100, C, H)`，含义完全不同，而且不会报错。

在本机 PyTorch 2.13 上实测：

```
pairwise_distance((4,8,1,1), (1,8,3,5)) -> (4, 8, 3)     # 而不是 (4, 3, 5)
```

本项目在 `src/spade/model.py` 里显式沿通道维计算，并用一条回归测试锁住语义，防止有人"顺手简化"回内置函数：

```
tests/test_model.py::test_channelwise_reduction_is_the_intended_distance
```

### 3.2 ImageNet 权重必须是 V1

torchvision 早期的 `pretrained=True` 等价于 `IMAGENET1K_V1`。后来新增的 `IMAGENET1K_V2` 是用改进配方重训的，特征分布不同，会让每一个数字都变。`features.py` 里硬编码 V1，并且拒绝其他骨干网络而不是悄悄降级。

### 3.3 常见实现会丢掉 gallery 的尾巴

```python
for d_idx in range(feat_gallery.shape[0] // 100):
    ...feat_gallery[d_idx * 100 : d_idx * 100 + 100]...
```

整除会丢掉最后 `G % 100` 行。K=5 时：`layer1` 15 680 → 用 15 600（丢 80）、`layer2` 3 920 → 3 900（丢 20）、`layer3` 980 → 900（丢 80）。

这显然是无意的，本项目默认使用完整 gallery。丢弃只会让最近邻距离变大（或不变），影响在小数点后第二位以内。想逐位对齐这个行为可以：

```python
SPADE(..., drop_gallery_remainder=True)
```

### 3.4 公开的数据下载地址已失效

`ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/...` 现在返回 `530 Login denied`（curl 退出码 67）。`scripts/prepare_data.py` 改成接受镜像 URL 或本地压缩包，并在解压后逐类核对图片数量。

---

## 4. 数据完整性校验

`scripts/prepare_data.py --check-only` 会核对 15 类的训练图数量：

| 类别 | train | 类别 | train | 类别 | train |
| --- | ---: | --- | ---: | --- | ---: |
| bottle | 209 | grid | 264 | screw | 320 |
| cable | 224 | hazelnut | 391 | tile | 230 |
| capsule | 219 | leather | 245 | toothbrush | 60 |
| carpet | 280 | metal_nut | 220 | transistor | 213 |
| | | pill | 267 | wood | 247 |
| | | | | zipper | 240 |

合计 3 629 张训练图、1 725 张测试图。数量对不上就说明镜像不完整，脚本以非零码退出。

---

## 5. 指标定义

- **图像级 ROC-AUC**：每张测试图一个分数，对 good / anomalous 标签算 AUC，再对 15 类取算术平均。
- **像素级 ROC-AUC**：把一个类别下**所有**测试图的全部像素摊平成一条序列，与真值掩码算 AUC，再对 15 类取平均。注意这是"按类别汇总后再平均"，不是"按图平均"。
- **PRO**（本项目额外提供）：按连通区域计算召回并对 FPR ≤ 0.3 积分归一化。像素级 AUC 会被大面积缺陷主导，PRO 让每个缺陷区域等权。

---

## 6. 已知的正常波动

- 与公开基准的差异在 ±0.5 个百分点以内属正常，来源是 PyTorch/torchvision 版本、BLAS 后端、`cdist` 的 matmul 展开路径带来的浮点差异。
- 本方法**无随机性**（不训练、不采样、DataLoader 不打乱），同一环境重复运行结果完全一致。实测：完整流水线跑第二遍，15 类的两个指标逐位相同。`config.seed` 只是卫生习惯，当前实现里没有任何一步会用到它。
- `grid` 类的图像级 ROC-AUC 只有 47.3%（低于随机），公开基准同样如此。这不是 bug，而是这套方法的已知短板：`grid` 的全局平均池化描述子无法区分正常与异常，原论文本身也未报告图像级分类结果。像素级定位反而很好（98.3%）。

---

## 7. 运行环境

每次运行的环境都记录在 `results.json` 的 `environment` 字段里（torch 版本、平台）。README 公布的数字在以下环境获得：

- Apple M2 / 16 GB，macOS
- PyTorch 2.13.0 + torchvision 0.28.0，MPS 后端
- Python 3.13

CPU 与 MPS 结果一致到小数点后一位；换设备只影响速度。

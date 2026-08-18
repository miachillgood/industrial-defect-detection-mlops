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

这显然是无意的，本项目**默认使用完整 gallery**。但"影响有多大"不该靠推理，所以两种模式都在 15 类上完整跑了一遍：

```bash
python -m spade.evaluate --categories all                              # 完整 gallery（默认）
python -m spade.evaluate --categories all --drop-gallery-remainder     # 复刻丢尾行为
```

像素级 ROC-AUC（%）三方对照：

| 类别 | 公开基准 README | 完整 gallery | 丢尾 | 哪个更近 |
| --- | ---: | ---: | ---: | :---: |
| bottle | 97.0 | 97.01 | 96.99 | 并列 |
| cable | 92.3 | 92.41 | 92.32 | 丢尾 |
| capsule | 98.4 | 98.39 | 98.38 | 完整 |
| carpet | 98.9 | 98.91 | 98.91 | 并列 |
| grid | 98.3 | 98.35 | 98.32 | 丢尾 |
| hazelnut | 98.5 | 98.53 | 98.52 | 丢尾 |
| leather | 99.3 | 99.32 | 99.32 | 并列 |
| metal_nut | 97.1 | 97.14 | 97.06 | 并列 |
| pill | 95.0 | 94.98 | 94.99 | 丢尾 |
| screw | 99.1 | 99.11 | 99.10 | 丢尾 |
| tile | 92.8 | 92.88 | 92.83 | 丢尾 |
| toothbrush | 98.8 | 98.85 | 98.83 | 丢尾 |
| transistor | 86.6 | 86.75 | 86.58 | 丢尾 |
| wood | 95.3 | 95.33 | 95.31 | 丢尾 |
| zipper | 98.6 | 98.59 | 98.55 | 完整 |
| **均值** | **96.4** | **96.44** | **96.40** | **丢尾** |

结论比预期更明确：

- **打开丢尾后，均值正好落在公开基准公布的 96.40**，偏差 0.00。
- 逐类平均绝对偏差从 **0.042 减半到 0.021**；15 类里 9 类更近、2 类更远、4 类并列。
- 我原先偏差最大的 `transistor`（0.15）在打开丢尾后降到 0.02——**之前把这 0.15 全归因于浮点路径差异是不对的，其中大部分来自这个 quirk**。
- **图像级完全不受影响**：两次运行 15 类的图像级 ROC-AUC 逐位相同（均值都是 85.41）。合理，因为图像级分数来自 avgpool 描述子，根本不经过 gallery。

两次运行的完整结果都在仓库里：`artifacts/runs/full-mvtec-k5/`（默认）与 `artifacts/runs/full-mvtec-k5-truncated/`（丢尾）。

**默认仍然保持完整 gallery**：更接近参考实现公布的数字不等于更正确，那 0.04 是被复刻的实现缺陷，不是精度提升。`--drop-gallery-remainder` 的意义是让"我们与基准的差异来自哪里"可以被审计，而不是拿来刷分。

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
- **PRO**（本项目额外提供）：按连通区域计算召回并对 FPR ≤ 0.3 积分归一化。像素级 AUC 会被大面积缺陷主导——一张图上漏掉一个小划痕，几乎不影响 AUC；PRO 让每个缺陷区域等权，漏掉小缺陷和漏掉大缺陷代价相同，更贴近产线关心的东西。公开基准与原论文都未报告该指标，所以没有对照值。

  PRO 默认打开。**这里我先前判断错过一次**：单类冒烟时看到开 PRO 后总耗时明显变长，就写成"大致翻倍"，还据此把默认设成关闭。后来用运行内的分段计时复核，发现那次变慢来自机器争用（当时还并行跑着别的实验），不是 PRO：

  | | 总耗时 | 特征提取 | 定位 | 其余（指标 + PRO + 可视化） |
  | --- | ---: | ---: | ---: | ---: |
  | 开 PRO | 19.5 min | 17.2 | 1.5 | **0.8** |
  | 关 PRO | 18.3 min | 15.7 | 2.1 | **0.5** |

  15 类合计 PRO 约 **18 秒**，占总耗时 2%（`per_region_overlap` 在 FPR 超过上限时会提前 break，不会真跑满 100 个阈值）。唯一的关闭理由既然不成立，默认就改成了打开。

  ```bash
  python -m spade.evaluate --categories bottle --compute-pro false    # 显式关闭
  ```

  之所以同时支持"裸开关"和"带值"两种写法，是因为 `dvc.yaml` 没有条件语法，必须无条件把 `--compute-pro ${evaluate.compute_pro}` 展开出来，于是这个参数必须能接受一个值。

---

## 6. 已知的正常波动

- 与公开基准的差异在 ±0.5 个百分点以内属正常，来源是 PyTorch/torchvision 版本、BLAS 后端、`cdist` 的 matmul 展开路径带来的浮点差异。
- 本方法**无随机性**（不训练、不采样、DataLoader 不打乱），同一环境重复运行结果完全一致。实测：完整流水线跑第二遍，15 类的两个指标逐位相同。`config.seed` 只是卫生习惯，当前实现里没有任何一步会用到它。
- `grid` 类的图像级 ROC-AUC 只有 47.3%（低于随机），公开基准同样如此。这不是 bug，而是这套方法的已知短板：`grid` 的全局平均池化描述子无法区分正常与异常，原论文本身也未报告图像级分类结果。

  值得注意的是**定位并没有跟着崩**：`grid` 的像素级 ROC-AUC 98.35%、PRO 86.39%，都在中上水平。也就是说两个阶段是解耦的——第一阶段的全局描述子挑不出"哪张图有问题"，但只要它挑出的 K 个近邻是合理的正常样本，第二阶段的逐位置对应照样能指出"问题在哪"。产线上如果图像级判定不可靠，可以改用像素级异常面积占比来做判定，`/predict` 返回的 `anomalous_pixel_ratio` 就是为此准备的。

- PRO 最低的两类是 `tile`（69.14%）和 `transistor`（70.76%），与它们像素级 ROC-AUC 也偏低（92.88% / 86.75%）一致。`transistor` 的缺陷常常是元件整体错位，"异常"在于全局布局而非局部纹理，逐位置最近邻对这种缺陷天然不敏感。

---

## 7. 运行环境

每次运行的环境都记录在 `results.json` 的 `environment` 字段里（torch 版本、平台）。README 公布的数字在以下环境获得：

- Apple M2 / 16 GB，macOS
- PyTorch 2.13.0 + torchvision 0.28.0，MPS 后端
- Python 3.13

换设备只影响速度，不影响结果：`bottle` 与 `grid` 两类在 CPU 上重跑，四个指标（97.22 / 97.01 / 47.28 / 98.35）与 MPS 逐位相同。

关于"完全一致"的一个例外：`artifacts/banks/*.pt` 的文件哈希每次构建都会变，因为元数据里带 `created_utc` 溯源时间戳。特征张量本身逐位相同——两次独立构建的 bank 做过 `torch.equal` 全量比对，唯一不同的键就是 `created_utc`。

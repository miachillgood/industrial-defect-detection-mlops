# MLOps 层

本文件描述检测算法之外的全部工程能力：数据与模型版本、实验记录、推理服务、人工审核闭环、容器化与 CI。

---

## 这套流水线要解决的实际问题

这套检测方法的部署形态和普通深度模型不一样，工程上有三个直接后果：

1. **没有权重文件。** "模型"= 冻结的 ImageNet 骨干 + 正常样本的特征库（memory bank）。所以**版本管理的对象是数据**，而不是 checkpoint —— 这正是 DVC 的用武之地。
2. **artifact 很大。** bottle 的 209 张训练图，特征库 fp16 下约 600 MB。绝不能进 git。
3. **阈值是产品决策，不是训练结果。** 判定阈值由业务对漏检/误检的容忍度决定，需要人工标注反馈来调 —— 这是 Streamlit 审核工具存在的理由。

```
prepare_data ──► evaluate ──► build_bank ──► FastAPI 服务
     │              │              │              │
    DVC           MLflow          DVC        Streamlit 审核
   (数据版本)     (实验记录)     (模型版本)      (人工反馈)
                                                  │
                                    阈值再标定 ◄───┘
```

---

## 1. DVC：数据与模型版本管理

### 为什么不是 git

`data/mvtec_anomaly_detection` 是 4.91 GiB、6 642 个文件（3 629 张训练图 + 1 725 张测试图 + 1 258 张真值掩码 + 少量说明文件）；`artifacts/banks/spade_bottle.pt` 561 MB。git 存不了这种东西。DVC 把内容哈希写进 `.dvc` 小文件由 git 跟踪，实际字节放到 remote。

### 常用命令

```bash
dvc status                 # 工作区与记录是否一致
dvc repro                  # 按 dvc.yaml 跑完整流水线
dvc metrics show           # 读 artifacts/runs/<run_name>/metrics.json
dvc metrics diff HEAD~1    # 与上一次提交比指标
dvc params diff            # 比 params.yaml
dvc dag                    # 打印阶段依赖图
```

当前仓库里 `dvc metrics show` 的输出：

```
Path                                       mean_image_rocauc  mean_pixel_rocauc  mean_pixel_pro  n_categories  delta_image_rocauc  delta_pixel_rocauc
artifacts/runs/full-mvtec-k5/metrics.json  85.41              96.44              86.13           15            0.01                0.04
```

### 特征缓存：DVC 管不到的那一层

DVC 的缓存粒度是**阶段**。改 `params.yaml` 里的 `top_k`，整个 `evaluate` 阶段重跑——包括那 15 分钟的特征提取。但特征只取决于（数据集、骨干、预处理），**跟 K 和 sigma 毫无关系**。扫一遍 K 就白白重算十几次完全相同的张量。

所以另加了一层阶段内缓存（`src/spade/cache.py`，默认关闭）：

```bash
python -m spade.evaluate --categories all --cache-features
```

实测 toothbrush 第二遍命中缓存，结果完全相同，特征耗时 56s → 20s（剩下的 20s 是未缓存的测试集）。

两个设计选择：

- **只缓存训练集**，和参考实现一样。训练集是 5 354 张里的 3 629 张，占大头；而缓存测试集还要连输入张量一起存（定位可视化需要），不划算。
- **缓存键覆盖所有能改变张量的因素**：类别、split、骨干、层集合、resize/crop、dtype、**torch 版本**。参考实现只用类名做键——改一下输入尺寸，它就会静默返回错误尺寸的特征。这条由 `tests/test_cache.py::test_every_input_dimension_changes_the_key` 锁住。

缓存全量约 20 GB（float32），已在 `.gitignore` 里，也不进 DVC——它完全可以从数据集重新生成，没有版本化的价值。

### README 结果表的自校验有个坑

README 的结果表由 `scripts/update_readme_results.py` 从 `results.json` 生成，CI 会用 `--check` 复核。最初 `--check` 的实现是"取 `artifacts/` 下 mtime 最新的 `results.json`"——加进第二个运行目录（丢尾对照实验）的当天就炸了：它去和消融实验对比，然后把完全正确的 README 报成 stale。

改成在结果块里写一行来源标记：

```html
<!-- source: artifacts/runs/full-mvtec-k5/results.json -->
```

`--check` 读这一行来决定跟谁比，README 因此自描述。回归测试 `tests/test_readme_sync.py::test_check_uses_the_declared_run_not_the_newest` 会同时验证"有标记时正确"和"去掉标记后确实会误报"，保证这个守卫不是摆设。

### git 与 DVC 的分工

`dvc.yaml` 里用 `cache: false` 明确划线：

- **git 管**：JSON 报告、`results.md`、ROC 曲线、15 类定位可视化——小、可读、要在页面上直接看到的证据；
- **DVC 管**：5 GB 数据集和 561 MB 特征库——只在 git 里留哈希。

### 流水线

`dvc.yaml` 定义三个阶段，依赖变了才重跑：

| 阶段 | 命令 | 输出 |
| --- | --- | --- |
| `check_data` | `scripts/prepare_data.py --check-only --strict` | — |
| `evaluate` | `python -m spade.evaluate` | `results.json` / `results.md` / `metrics.json` / `roc_curve.png` / `images/` |
| `build_bank` | `scripts/build_bank.py` | `spade_<category>.pt` + 标定元数据 |

改 `params.yaml` 里的 `evaluate.top_k`（比如从 5 改成论文的 50），`dvc repro` 只会重跑 `evaluate` 和 `build_bank`，不会重新校验数据。

一个 DVC 的实际约束：`dvc.yaml` 的 `cmd` 没有条件语法，`${evaluate.compute_pro}` 只能被无条件展开。所以 `--compute-pro` 被实现成"既能当裸开关、也能带值"（`--compute-pro` / `--compute-pro false`），而不是普通的 `store_true`。这条约束由 `tests/test_cli.py::test_dvc_style_invocation_parses` 锁住。

### remote

仓库里默认配置的是仓库内的本地 remote（`.dvcstore`，已在 `.gitignore` 里），克隆下来就能用。生产换成对象存储只改一行：

```bash
dvc remote add -d store s3://my-bucket/spade     # 或 gs:// / azure:// / ssh://
dvc push
```

macOS 的 APFS 支持 reflink，`.dvc/config` 里配了 `cache.type = reflink,hardlink,copy`，所以把 5 GB 数据集纳入 DVC 只花了 8 秒、几乎不占额外磁盘。

---

## 2. MLflow：实验记录

`src/mlops/tracking.py` 是一层薄封装，设计上**失败不影响主流程**：MLflow 装没装、服务通不通，`make eval` 都必须能跑完并写出 `results.json`。追踪是观测手段，不是运行依赖。

每次运行记录：

- **params**：`top_k`、`backbone`、`resize`/`cropsize`、`device`、torch 版本、平台
- **metrics**：15 个类各自的 `image_rocauc` / `pixel_rocauc`，以及 `mean_image_rocauc` / `mean_pixel_rocauc` / `wall_clock_s`
- **artifacts**：整个运行目录（对照表、ROC 曲线、定位可视化）

```bash
make mlflow                                    # UI -> http://localhost:5000
MLFLOW_TRACKING_URI=http://localhost:5000 make eval
```

默认后端是 `sqlite:///mlflow.db`，本地文件数据库，不需要起服务。

> **踩过的坑：** 最初默认用的是 `file:./mlruns`。MLflow 3.15 把纯文件后端置为 maintenance mode，现在会直接抛异常（除非设 `MLFLOW_ALLOW_FILE_STORE=true`）。这次刚好验证了 fail-soft 的设计是对的——追踪初始化失败时，15 类的评估照常跑完并写出了 `results.json`，只在日志里留下一行 `[spade] MLflow disabled (...)`。事后用 `log_existing_run()` 把那次运行补录进去即可，不用重跑 21 分钟。

离线跑完再补记录：

```python
from mlops.tracking import log_existing_run
log_existing_run("artifacts/runs/full-mvtec-k5/results.json")
```

---

## 3. FastAPI：推理接口

```bash
make api        # http://localhost:8000/docs
```

| 端点 | 用途 |
| --- | --- |
| `GET /health` | 存活探针 + 已加载的类别 |
| `GET /models` | 每个 memory bank 的元数据与标定信息 |
| `GET /stats` | 请求数、异常判定数、平均时延 |
| `POST /predict?category=bottle` | 返回分数、判定、base64 热力图与掩码 |
| `POST /predict/overlay?category=bottle` | 直接返回叠加后的 PNG |

```bash
curl -s -F 'file=@data/mvtec_anomaly_detection/bottle/test/broken_large/000.png' \
  'http://localhost:8000/predict?category=bottle&include_images=false' | jq
```

实测（bottle，阈值 6.78）：

| 输入 | image_score | 判定 | 异常像素占比 |
| --- | ---: | --- | ---: |
| `test/good/000.png` | 4.91 | OK | 0.28 % |
| `test/broken_large/000.png` | 10.57 | DEFECT | 23.4 % |

`POST /predict/overlay` 直接返回叠加图：

![API overlay](assets/api_overlay_bottle_broken_large_000.png)

设计要点：

- **按需加载。** 每个类别的 bank 首次请求时才载入并缓存，加锁保证线程安全。`SPADE_EAGER_LOAD=1` 可改为启动时全部载入（容器里配合就绪探针更合适）。
- **bank 不进镜像。** 它是数据，由 DVC 管理，通过 volume 挂载。镜像里只烘焙 ImageNet 权重，保证冷启动不联网。
- **缺 bank 返回 404 而不是崩溃。** 错误信息直接告诉你去跑 `scripts/build_bank.py`。

### 阈值怎么来的

`scripts/build_bank.py` 在**训练集内部**做留一法：把每张训练图当查询，排除自己后取 K 近邻算分，得到"正常样本分数分布"，再取分位数作阈值（默认图像级 P99、像素级 P99.5）。**全程不碰测试集**，所以报告的 ROC-AUC 没有被阈值标定污染。

---

## 4. Streamlit：标注审核工具

```bash
make review     # http://localhost:8501
```

人在回路的界面：模型给判定和缺陷掩码，质检员确认或推翻，每条决定追加进 `artifacts/reviews/reviews.jsonl`。

**审核队列**：并排显示输入图 / 热力图 / 预测掩码 / 真值掩码（测试集样本才有），给出分数、阈值、模型判定与异常像素占比；质检员选 `ok` / `defect` / `unsure`，可填缺陷类型和备注。侧边栏的阈值滑块可以实时看判定怎么随阈值变化 —— 这正是这个工具的核心价值。

**看板**：累计审核量、人机一致率、以人工判定为真值的混淆矩阵与 precision/recall/F1，以及"分歧列表"——这些就是重新标定阈值的依据。可导出 CSV。

存储用 append-only JSONL：

- 多人同时审核不会互相覆盖
- 改判是新增一条记录，不是就地修改，历史完整保留
- 尾部写坏一行不会毁掉整个文件（`load()` 会跳过解析失败的行）

`latest_per_image()` 按图片路径取最后一条决定作为当前状态。

---

## 5. Docker

`docker/Dockerfile` 一个多阶段构建，两个目标：

```bash
docker build -f docker/Dockerfile --target api    -t spade-api .
docker build -f docker/Dockerfile --target review -t spade-review .
docker compose -f docker/docker-compose.yml up --build    # api + review + mlflow
```

几个刻意的选择：

- **CPU 版 torch**（`--index-url https://download.pytorch.org/whl/cpu`）。默认源的 wheel 会拖进约 2 GB CUDA 库，在 CPU 容器里毫无用处。
- **烘焙 ImageNet 权重**，冷启动不需要联网。
- **非 root 用户**（uid 10001）。
- **HEALTHCHECK** 打到各自的健康端点。
- **bank 只读挂载**，不进镜像。

---

## 6. GitHub Actions

`.github/workflows/ci.yml` 三个 job：

| Job | 内容 |
| --- | --- |
| `lint` | `ruff check .` |
| `test` | Python 3.11 / 3.12 矩阵，跑 `pytest -m "not needs_data"`，缓存 torch hub 权重，产出覆盖率 |
| `docker` | 构建 API 镜像并冒烟：起容器 → `/health` 就绪 → 无 bank 时 `/predict` 必须返回 404 |

CI 里没有 5 GB 数据集，所以需要数据的测试用 `needs_data` 标记跳过；需要骨干网络的测试用 `needs_backbone` 标记，权重走 cache。

`tests/test_docs_attribution.py` 把署名要求变成了会失败的测试：README 必须提到论文、作者、`byungjae89/SPADE-pytorch`、`NVlabs/SPADE` 的消歧说明、以及 `K=5` 与 `K=50` 的区别。任何文档若把这个第三方实现描述成论文作者发布的版本，CI 直接挂掉。

这条检查带一个"守卫的守卫"：`test_the_attribution_guard_actually_catches_a_bare_claim` 保证正则本身没被改松。文档里允许**引用**错误说法来做反面警示（靠上下文里的否定词识别），但不允许直接这么断言。

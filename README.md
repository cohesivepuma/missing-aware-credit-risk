# missing-aware-credit-risk

**面向不完整数据的缺失机制感知与概率校准个人信用风险评估方法**

本仓库用于课程科研项目和论文实验，目标是建立结构清晰、可多人协作、可复现的个人信用风险研究框架。第一阶段最小框架已完成，目前已接入公开 German Credit 数据，支持固定划分和 Logistic Regression；高级研究模块将在后续 Issues 中逐步完成。

Toy 指标只用于检查流程，不能作为信用风险研究结论。

## 研究问题与三个模块

1. **Missingness Benchmark**：在公开信用数据上构造 MCAR、MAR、MNAR，比较 10%、30%、50% 缺失率下的判别能力与概率质量。机制参数、掩码生成方式和实际缺失率需要明确记录。
2. **Mask-aware Modeling**：比较 Logistic Regression、Random Forest、LightGBM、普通 MLP 与 Mask-aware MLP，研究缺失模式本身是否提供额外信息。
3. **Probability Calibration**：在独立验证集上拟合 Platt Scaling、Isotonic Regression 或 Temperature Scaling，研究缺失情况下的概率可靠性及校准收益。

核心问题是：不同缺失机制如何影响模型？显式输入缺失掩码能否提升鲁棒性？校准后能否得到更可靠的违约概率？

## 当前实现状态

| 功能 | 当前状态 |
| --- | --- |
| Python / NumPy / PyTorch 统一种子 | 已实现 |
| Toy 数据与本地数值 CSV 加载 | 已实现 |
| 官方 German Credit 获取、校验与质量检查 | 已实现 |
| 分层 train / validation / test 划分 | 已实现，默认 60% / 20% / 20% |
| 共享行索引与独立验证子集 | German Credit 默认 train/tune/calibration/test = 600/100/100/200 |
| 训练集拟合的中位数填充与标准化 | 已实现 |
| 训练集拟合的类别众数填充与 one-hot | German Credit 已实现 |
| Logistic Regression baseline | 已实现 |
| AUC、F1、Brier Score、ECE | 已实现 |
| 配置驱动运行与本地 JSON 报告 | 已实现 |
| pytest 与 GitHub Actions | 已配置 |
| MCAR / MAR / MNAR | 已实现，支持固定 seed、精确缺失率、训练集拟合计划和可复用 mask |
| LightGBM、MLP、Mask-aware MLP | 接口与 TODO，尚未训练实现 |
| Random Forest | 后续计划，当前未建立实现 |
| Platt Scaling / Isotonic Regression 概率校准 | 已实现，只在 validation_calibration 拟合 |
| 校准前后配对对比与可靠性图 | 已实现，各变体使用同一批测试样本 |
| 缺失实验、消融实验 | 接口与 TODO，尚未实现 |
| Temperature Scaling | 仅定义 logits adapter，温度拟合尚未实现 |
| Streamlit Demo | 状态展示入口，尚无预测功能 |

## 环境安装

使用 **Python 3.11**；本地验证版本为 3.11.13，记录在 `.python-version`。直接依赖和 SciPy 求解器依赖版本固定在 `requirements.txt`，PyTorch 与 sklearn 版本信息可参考 [PyTorch 2.6.0](https://pypi.org/project/torch/2.6.0/) 和 [scikit-learn 1.6.1](https://pypi.org/project/scikit-learn/1.6.1/)。

```bash
git clone https://github.com/cohesivepuma/missing-aware-credit-risk.git
cd missing-aware-credit-risk
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 激活命令：`.venv\Scripts\activate`。如使用 uv，也可先执行 `uv python install 3.11.13` 和 `uv venv --python 3.11.13 --seed .venv`，随后仍使用上述 pip 安装命令。

当前 baseline 不导入 LightGBM；后续在 macOS 使用 LightGBM 时可能需要 OpenMP 运行库，参见其[官方安装说明](https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html)。CI 使用同版本的 CPU PyTorch wheel，避免下载不需要的 CUDA 依赖。

## 数据与模型接口

数据加载和划分模块使用同一结构：

```python
data = {
    "x": ...,     # float64, shape (n_samples, n_features)，原始缺失值是 NaN
    "mask": ...,  # uint8, 同 x 的形状；1 = observed，0 = missing
    "y": ...,     # int64, shape (n_samples,)，1 = 正类/坏信用，0 = 负类/好信用
}
```

**mask 始终是 `1 = observed`、`0 = missing`。** 填充后的值不会使原来的 missing mask 变成 observed。当前 toy 数据全部 observed，标签仅是合成二分类标签。

German Credit 的 x 保留原始 20 个字段，类别代码是数值载体，模型内部仅在训练集拟合 one-hot。其标签是 Good/Bad 信用分类，不解释为特定期限违约事件。缺失注入在 one-hot 前进行，详见 [数据协议](docs/data_protocol.md)。

```python
model.fit(train["x"], train["y"], mask=train["mask"])
p = model.predict_proba(test["x"], mask=test["mask"])[:, 1]
metrics = evaluate_metrics(test["y"], p)
# {"auc": float, "f1": float, "brier": float, "ece": float}
```

模型输出 `(n_samples, 2)`，列顺序固定为 `[P(y=0), P(y=1)]`。Logistic Regression 使用特征值，忽略 mask。深度学习 adapter 后续遵循相同外部接口。校准接口接收与返回一维 `P(y=1)`；Temperature Scaling 将单独约定 logits 接口。

AUC 和 F1 越大越好，Brier Score 和 ECE 越小越好。默认 F1 阈值为 0.5，`p >= threshold` 判为正类。ECE 使用 10 个等宽概率区间，按样本占比加权各区间的平均正类概率与正类频率之差；最后一个区间包含 1，空区间贡献为 0。这是正类概率 ECE，报告时必须同时记录分箱数。

## 项目目录

```text
missing-aware-credit-risk/
├── README.md                 # 项目说明与复现实验入口
├── requirements.txt          # 固定版本的直接依赖
├── .python-version           # Python 版本
├── .gitignore                # 排除数据、权重、结果与缓存
├── pyproject.toml            # pytest 路径配置
├── CONTRIBUTING.md           # 多人协作约定
├── CLAUDE.md                 # AI 协作代码约定
├── .github/
│   ├── workflows/tests.yml   # Python 3.11 测试与 baseline 检查
│   └── pull_request_template.md
├── configs/
│   ├── baseline.yaml         # 当前可运行配置
│   ├── german_credit.yaml    # 真实信用数据与共享划分配置
│   ├── calibration.yaml      # 冻结基础模型后的校准对比配置
│   └── experiment.yaml       # 研究矩阵与缺失机制参数，统一 runner 仍待集成
├── data/
│   └── README.md             # 公开数据获取与本地数据约定
├── scripts/
│   ├── prepare_german_credit.py # 官方下载、转换、质量报告、行索引
│   └── inspect_german_credit.py # 五行轻量预览
├── docs/
│   ├── data_protocol.md      # 字段协议及课程报告数据/预处理正文
│   ├── calibration_protocol.md # 校准协议、结果产物及课程报告校准正文
│   └── missingness_protocol.md # MCAR/MAR/MNAR 定义、复现和报告协议
├── src/
│   ├── __init__.py
│   ├── data/                 # loader.py / preprocess.py / german_credit.py / prepare.py / missing_generator.py
│   ├── models/               # base.py / logistic.py / lightgbm_model.py / mlp.py / mask_aware_mlp.py
│   ├── calibration/          # calibrators.py / reliability.py
│   ├── metrics/              # metrics.py
│   └── utils/                # seed.py
├── experiments/
│   ├── run_baseline.py       # 可运行的训练与评价流程
│   ├── run_calibration.py    # 校准前后配对对比与可靠性图
│   ├── run_missingness.py    # TODO
│   └── run_ablation.py       # TODO
├── results/
│   └── README.md             # 实验产物约定
├── app/
│   └── streamlit_app.py      # 可选 Demo 入口
└── tests/
    ├── test_data.py          # 标签、来源校验、共享划分和类别拟合边界
    ├── test_calibration.py   # 校准边界、输入校验与拟合/预测分离
    ├── test_reliability.py   # 分箱协议一致性与可靠性图输出
    ├── test_missing_generator.py
    └── test_metrics.py
```

所有 `src` 子包包含 `__init__.py`。`data/raw/`、`data/processed/`、`checkpoints/` 和实际实验结果仅在本地生成，不提交 Git。

## 运行 baseline

在仓库根目录、激活虚拟环境后执行：

```bash
python experiments/run_baseline.py
```

无需下载数据。脚本生成 1000 行、12 个特征的 toy 数据，用训练集拟合填充、缩放和 Logistic Regression，然后打印测试集 AUC / F1 / Brier / ECE，保存到 `results/baseline.json`。验证集被单独保留，本阶段不用于拟合或评价。

自定义配置和输出：

```bash
python experiments/run_baseline.py --config configs/baseline.yaml --output results/my_run.json
```

配置中的 CSV 路径和输出路径相对于仓库根目录；显式传入的 `--config` 路径相对于当前工作目录。输入数据说明见 [data/README.md](data/README.md)。输出包含配置、种子、数据指纹、划分大小、环境版本和指标，见 [results/README.md](results/README.md)。

运行真实信用数据：

```bash
python scripts/prepare_german_credit.py
python experiments/run_baseline.py --config configs/german_credit.yaml
```

首次准备从官方来源下载并校验，后续复用本地缓存。处理后的 CSV、元数据、质量报告与 splits_seed42.json 保存在被 Git 忽略的 data/processed/german_credit/。同一索引文件被所有模型复用，并核对数据指纹和划分配置。German Credit 验证集拆分为互斥的 validation_tune 与 validation_calibration，父视图 validation 不单独计入样本总数。

## 运行概率校准对比

先训练并冻结基础模型，再用独立的 validation_calibration 拟合校准，最后在同一批测试样本上比较校准前后：

```bash
python experiments/run_calibration.py
python experiments/run_calibration.py --config configs/german_credit.yaml --output results/german_credit_calibration.json
python experiments/run_calibration.py --seeds 42 43 44 --output results/calibration_multiseed.json
```

默认配置使用合成数据，无需下载任何数据。报告写入 `results/calibration.json`，可靠性图写入 `results/`。对比是配对的：`uncalibrated`、`platt`、`isotonic` 使用完全相同的测试行、相同阈值与相同分箱数；测试集不参与选择校准方法、阈值或分箱数。校准可能改善也可能退化，报告如实记录 `delta_vs_uncalibrated`，不把指标必然下降作为验收条件。多种子模式要求 `split.manifest_path` 为包含 `{seed}` 的模板。详见[校准协议](docs/calibration_protocol.md)。

## 测试与可复现性

```bash
python -m pytest
```

测试覆盖手算指标、概率边界、mask 约定、三个随机数生成器、固定种子划分与预测、划分互斥、缺失值填充和训练集预处理边界。缺失机制测试进一步覆盖精确缺失率、输入不变、已有缺失保留、NaN/mask 对齐、训练集参数复用、标签独立、MAR 驱动可观测和 MNAR 特征相关。校准测试覆盖概率边界、输入校验、拟合/预测分离、映射方向、校准参数记录与可靠性分箱协议一致性。

`set_seed(seed)` 设置 Python random、NumPy、PyTorch 和 CUDA 种子，同时请求确定性 PyTorch 运算。Toy 生成和两次分层划分均显式使用同一种子。Logistic Regression 的中位数填充与标准化只通过训练集 `fit`，测试集仅 `predict_proba`。

同配置、同环境下应得到相同划分与预测。跨硬件、操作系统和数值库不承诺逐位相同。直接依赖已固定，但传递依赖未完全锁定；每次运行记录环境版本。若研究需要精确归档环境，可在本地执行 `python -m pip freeze > results/environment.txt`。Python 的 hash 种子必须在进程启动前设置才生效，如 `PYTHONHASHSEED=42 python experiments/run_baseline.py`；实验不依赖 hash 顺序。

GitHub Actions 在 push 和 Pull Request 上执行 pytest，并重复运行 baseline 与校准对比两次以比较报告是否逐字节一致。

## 后续实验计划

1. 已接入 German Credit，并完成目标映射、特征元数据、官方来源校验、固定行索引和训练集类别处理；后续数据集按相同协议扩展。
2. 已实现并验证 MCAR / MAR / MNAR。先划分，再在训练集拟合机制阈值，再向各划分应用缺失，最后填充。统一实验 runner 的集成由成员 3 负责。
3. 完成 Random Forest、LightGBM、普通 MLP；在相同划分、缺失掩码与预算下比较 baseline。
4. 实现 Mask-aware MLP，并开展 with / without mask 消融。
5. 已在独立验证子集 validation_calibration 实现 Platt Scaling 与 Isotonic Regression 校准、配对的前后对比和可靠性图；Temperature Scaling 单列 logits adapter，温度拟合仍待实现。
6. 对三种机制、三个缺失率、多个种子运行配对实验，报告均值与标准差、可靠性图和校准前后对比。
7. 研究结果确认后再实现简单 Streamlit 输入与预测展示。

协作流程、PR 验证要求和建议的首批 Issues 见 [CONTRIBUTING.md](CONTRIBUTING.md)。

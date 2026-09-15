# 概率校准与可靠性分析协议

关联任务：[成员 5 / Issue #5](https://github.com/cohesivepuma/missing-aware-credit-risk/issues/5)。本文定义基础模型冻结后的概率校准边界、评价协议与结果产物；代码实现和团队评审以该 Issue 对应 PR 为准。

## 可用于课程报告的正文

本研究在基础模型训练完成后冻结其全部参数，再使用与调参、早停和测试完全独立的校准验证子集拟合概率校准器。这样处理的原因是：判别能力与概率可靠性是两个不同目标，判别力较强的模型仍可能给出系统性偏高的正类概率；而如果在测试集上选择校准方法、阈值或分箱数，报告的校准收益就会包含测试集信息，失去最终评价的意义。

本研究实现两种校准方法。Platt Scaling 在概率的对数几率（logit）尺度上拟合一个几乎无正则化的 Logistic 回归，得到 `sigmoid(a * logit(p) + b)`；斜率 a 由独立校准子集拟合，可以为正、零或负。Isotonic Regression 使用 sklearn 的单调非减映射，在拟合阈值之间线性插值，平坦区间会产生并列概率；输出被限制在 [0, 1] 区间，但在校准样本较少时更容易过拟合。为避免概率恰好为 0 或 1 时 logit 发散，Platt Scaling 在变换前将概率裁剪到 [ε, 1 − ε]，本实验取 ε = 1e-6，对应 |logit| ≤ 13.9。自定义 ε 必须保证 float64 中 `1−ε < 1`，过小而不能区分上界的参数会被拒绝。裁剪可能使极端概率成为并列值。

校准前后使用完全相同的测试样本、相同的分类阈值（0.5）与相同的分箱数（10），指标定义与全项目一致：AUC 与 F1 越大越好，Brier Score 与 ECE 越小越好。ECE 按等宽概率区间计算，按样本占比加权各区间的平均正类概率与正类频率之差，最后一个区间包含概率 1，空区间贡献为 0；这是正类概率 ECE，报告时必须同时记录分箱数。需要强调的是，本文不把“Brier 与 ECE 在每次实验中均下降”作为验收条件：校准可能改善概率质量而略微降低排序指标，也可能因校准样本有限而退化。我们如实记录每次实验的实际改善或退化。

可靠性图以横轴表示各分箱的平均预测正类概率，纵轴表示该分箱中实际的正类频率，同时给出理想对角线、每个分箱的样本数以及各曲线的 ECE。曲线贴近对角线说明概率更可靠，偏离对角线说明存在系统性高估或低估；样本数较少的箱在解释时需要更谨慎。本图的理想线、分箱数与测试样本数均在图中标明。

Temperature Scaling 面向直接输出 logits 的深度模型，需要一个单独的 logits 适配器（1 维 logits、保持行顺序、单独拟合并固定温度）。该适配器在本次变更中只定义接口与数值变换，未实现温度拟合，因此本文不报告任何 Temperature Scaling 结果。

## 结果解读规则

配对结果必须分开读，否则很容易得出错误结论：

- **AUC 不变需要条件。** Platt 的斜率 a 为正，且裁剪、浮点舍入或 sigmoid 饱和没有引入新的并列值时，样本排序保持不变，AUC 不变。a 为负会反转排序，a 为零会输出常数；即使 a 为正，裁剪产生的新并列值也可能改变 AUC。因此 AUC 变化本身不等于实现错误，应结合拟合斜率和并列概率解释。Isotonic 的平坦区间也会产生并列值，AUC 可以升高或降低。
- **F1 需要结合阈值解释。** 可以在同一个预先设定的阈值（本流程为 0.5）下配对报告校准前后的 F1，说明校准改变了哪些分类决策；概率可能上移或下移，F1 的方向并不固定。若要报告优化阈值后的 F1，应对每个变体仅在 `validation_tune` 上选择阈值，再固定阈值用于测试。不要把固定阈值与优化阈值两种协议混为一谈，也不要在测试集上搜索阈值。本流程当前只报告固定阈值结果。
- **校准样本有限时，Brier 与 ECE 可能退化。** 校准器只有 `validation_calibration` 的样本可用于拟合；当该子集较小、而基础模型本身已接近校准时，加入校准反而会引入估计噪声。这是需要如实报告的结果，不是需要掩盖的失败。
- 因此结论应表述为“在给定划分、给定校准样本量与给定阈值协议下，校准改善/退化/无明显影响”，而不是泛化的“校准更好”。

## 范围与数据边界

- 校准器只在 `validation_calibration` 上拟合，不使用 `validation_tune`、不使用测试集统计量，也不使用标签之外的任何测试信息。
- 基础模型在校准前冻结：`experiments/run_calibration.py` 在同一进程中对训练集调用一次 `fit`，之后不再重拟合、不重调超参数、不替换模型。
- 校准只在基础模型已完成训练后进行；允许在校准子集上拟合的只有校准器自身的参数。
- 校准接口接收与返回一维 `P(y=1)`，不接收 `(n_samples, 2)` 概率矩阵，也不改变模型的判别阈值语义。

## 接口契约

```python
from src.calibration.calibrators import ProbabilityCalibrator

calibrator = ProbabilityCalibrator("platt", clip_epsilon=1e-6)
calibrator.fit(validation_probabilities, validation_labels)   # 只允许校准子集
calibrated = calibrator.predict_proba(test_probabilities)     # 一维 P(y=1)，与输入行顺序一致
```

- `fit(probabilities, y)` 要求一维、等长、概率在 [0, 1]、标签为 0/1，且两个类别都存在。
- `predict_proba(probabilities)` 必须在 `fit` 之后调用，否则抛出 `RuntimeError`；返回值为 `float64` 一维数组，且始终位于 [0, 1]。
- `is_fitted` 与 `fitted_parameters` 用于报告与调试；`fitted_parameters` 在未拟合时同样抛出 `RuntimeError`。
- Isotonic 关于输入概率单调非减；Platt 的单调方向由拟合斜率 a 决定，输出裁剪与数值舍入可能产生并列概率。

`LogitsAdapter` 固定 logits 契约：一维 logits、与 `P(y=1)` 相同的行顺序、稳定 sigmoid、裁剪后的 logit 变换，以及给定温度下的 `sigmoid(logits / T)`。`LogitsAdapter.fit_temperature` 目前显式抛出 `NotImplementedError`，表示温度拟合属于后续扩展，而不是已实现的方法。

## 指标与分箱协议

- 四项指标统一由 `src/metrics/metrics.py` 的 `evaluate_metrics` 返回，键固定为 `auc`、`f1`、`brier`、`ece`。
- 标签编码固定为 0/1；概率必须有限且位于 [0, 1]。该校验由 `validate_binary_inputs` 与 `validate_probabilities` 统一提供，校准器、指标和可靠性图共用同一套检查。
- 正向判定固定为 `P(y=1) >= threshold`，默认 0.5；阈值选择属于验证集工作，本模块不提供测试集阈值搜索。
- 分箱定义统一由 `assign_probability_bins` 提供：等宽区间，除最后一箱外均为左闭右开，最后一箱额外包含概率 1。ECE 与可靠性图调用同一个函数，因此图中的分箱与报告中的 ECE 一定对应同一协议。
- 校准前后必须使用同一批测试样本，禁止为了凑出更好看的结果更换评价子集。
- 绘图函数会拒绝各曲线标签长度或标签顺序不一致的输入；样本身份的一致性由实验 runner 复用同一个 test 划分保证。

## 结果产物

`python experiments/run_calibration.py` 默认写入 `results/calibration.json`（被 Git 忽略），并生成 `results/reliability_<数据名>_seed<种子>.png`。报告字段：

| 字段 | 含义 |
| --- | --- |
| `config` | 完整 YAML 配置，含种子、数据、划分、模型、校准与评价设置 |
| `seeds` / `runs` | 实际运行的种子列表与每次运行的完整结果 |
| `runs[].dataset_sha256` | 输入 x / mask / y 的指纹，用于识别相同输入 |
| `runs[].split_sizes` | train / validation / validation_tune / validation_calibration / test 样本数 |
| `runs[].calibration` | 拟合划分、拟合样本数、评价划分、评价样本数、方法、裁剪值与阈值、分箱数与已拟合参数 |
| `runs[].metrics` | `uncalibrated` 与各校准方法在同一批测试样本上的四项指标 |
| `runs[].delta_vs_uncalibrated` | 各方法相对未校准的指标差值；Brier/ECE 的负值表示改善 |
| `runs[].reliability_diagram` | 该次运行生成的可靠性图路径，未生成时为 null |
| `summary` | 多种子运行时各变体指标的均值与标准差；单种子运行时为 null |
| `environment` | Python、平台与核心包版本 |

报告不含随机时间戳，同配置、同环境下重复运行应得到逐字节相同的 JSON，可直接用于复现检查。

## 复现命令

```bash
# 合成数据默认流程，无需下载任何数据
python experiments/run_calibration.py

# 真实 German Credit 数据；先执行 python scripts/prepare_german_credit.py
python experiments/run_calibration.py --config configs/german_credit.yaml \
    --output results/german_credit_calibration.json

# 多种子汇总；需要 split.manifest_path 为包含 {seed} 的模板
python experiments/run_calibration.py --seeds 42 43 44 \
    --output results/calibration_multiseed.json
```

多种子模式下，换种子会改变划分，因此 `split.manifest_path` 必须写作包含 `{seed}` 的模板（例如 `data/processed/german_credit/splits_seed{seed}.json`），否则脚本会明确拒绝运行，避免复用与当前种子不匹配的共享行索引。

输出路径优先取显式 `--output`，其次取 `output.calibration_path`。如果配置只有 baseline 使用的 `output.path`，脚本会在文件名后缀前添加 `_calibration`，例如 `results/german_credit_baseline.json` 对应 `results/german_credit_baseline_calibration.json`，保留原 baseline 报告。可靠性图默认保存在校准报告所在目录。`calibration.methods` 中的 `none` 表示“不校准”这一对照，每次运行都会自动包含 `uncalibrated`，因此 `none` 无需重复列出；至少需要一个实际校准方法。

## 实验影响说明

- 本变更不改变数据划分、mask 约定、随机种子语义与指标定义；`assign_probability_bins` 只是把原有的分箱实现抽出为公共函数，ECE 数值与之前完全一致。
- 不改变任何模型的外部接口：校准器作用于模型输出的 `P(y=1)`，模型仍返回 `(n_samples, 2)`。
- 不新增数据依赖，不改变已有配置文件的默认行为；`configs/calibration.yaml` 是新增文件。

## 对接约定

- **成员 3（集成负责人）**：`experiments/run_calibration.py` 的报告包含配置、数据指纹、划分大小、指标和环境版本；三种变体在同一批测试样本上配对比较。接入统一实验流程时请复用 `runs[].metrics` 字典，不要重新定义指标名。
- **成员 4（深度模型）**：Mask-aware MLP 只需按现有外部接口提供 `(n_samples, 2)` 概率，校准器即可直接作用于第 1 列 `P(y=1)`。若模型暴露 logits，请使用 `LogitsAdapter` 的契约；在本次变更中不要依赖未实现的温度拟合。有/无校准的消融入口由 `calibration.methods` 控制，不额外增加参数。
- **成员 5**：负责校准器、可靠性图与本节报告正文；不根据测试结果选择校准方法、阈值或分箱数。多种子汇总需要其他模型负责人提供各自的 `results/` 报告后合并。

未来若需要加入 Temperature Scaling，应新增 logits 拟合实现、相应测试与报告段落，并通过单独 PR 同步本文与 README 的状态表。

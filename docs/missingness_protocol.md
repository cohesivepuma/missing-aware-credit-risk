# MCAR、MAR 与 MNAR 缺失机制协议

关联任务：[成员 2 / Issue #2](https://github.com/cohesivepuma/missing-aware-credit-risk/issues/2)。本文定义可用于课程报告和统一实验的缺失注入语义。

## 可用于课程报告的正文

本研究在固定数据划分之后、填充和类别 one-hot 编码之前注入人工缺失。缺失仅作用于原始字段矩阵 `x` 和同形掩码 `mask`，标签 `y` 与原始行顺序保持不变。掩码统一采用 `1 = observed`、`0 = missing`。每次注入都先复制输入，因此不会修改共享原始数据，也不会覆盖数据划分阶段已存在的自然缺失。

设允许注入的特征集合为 `E`，其中当前已观测单元为候选集合 `C={(i,j): j in E, mask[i,j]=1}`。参数 `rate` 定义为本次新增缺失占候选集合的比例。实现使用无放回抽样精确选择 `round(rate * |C|)` 个单元，并记录请求缺失率、实际新增缺失率、候选单元数、注入单元数和注入后的整体缺失率。由于中文类别代码、列顺序和模型均不被用于随机选择，相同数据、机制、参数和随机种子会得到相同的缺失位置。

MCAR 假设每个候选单元具有相同的缺失概率，缺失选择不依赖任何特征值或标签。MAR 根据始终可观测的驱动变量决定候选单元的相对缺失概率；驱动列从可注入集合中移除，因此不会在本次注入中变为缺失。MNAR 根据单元注入前的自身特征值决定相对缺失概率。MAR 和 MNAR 的分位点只从训练集拟合，冻结后应用到训练、验证和测试划分，避免验证集或测试集统计量进入缺失机制。概率权重仅用于无放回抽样的相对排序，最终新增缺失数量始终由 `rate` 精确控制。

类别载体代码不表示连续大小关系。因此 MNAR 的实验范围应限制为数值字段或具有明确顺序的字段；对于无序类别字段，如果研究需要 MNAR，应单独定义类别条件概率并通过新的机制说明和测试评审，不能直接把官方枚举代码当作连续值。

MAR 驱动列在训练集以及每个待应用的划分中，都必须已经满足 `mask=1` 且数值有限。`fit_missingness_plan` 和 `apply_missingness` 均检查全部驱动列；部分或全部 NaN、正负无穷会触发 `ValueError`，即使 `rate=0` 或 `rate=1` 也如此。调用者应选择各划分均完整观测的驱动字段；缺失生成器不会填充驱动值、把缺失标记改为观测，或回退到均匀抽样。可注入的目标列仍允许自然缺失，并保留原有缺失单元。

## 三种机制

| 机制 | 相对权重来源 | 明确不使用 | 关键约束 |
| --- | --- | --- | --- |
| MCAR | 所有候选单元相同 | 特征值、标签、列类型 | 固定 seed 后均匀、无放回选择 |
| MAR | 训练集上冻结的可观测驱动变量分位点 | 候选单元自身值、标签 | 驱动列保持 `mask=1` |
| MNAR | 训练集上冻结的候选列自身分位点 | 标签 | 默认仅限数值或有序字段 |

默认 `direction="higher"` 表示数值越大，缺失相对权重越高；`direction="lower"` 反向。`strength` 控制机制信号强度，权重形式为 `1 + strength * score`，其中 `score` 位于 `[0,1]`。`strength` 不改变目标缺失数量。

## 划分与拟合边界

正确流程如下：

1. 读取原始数据并建立固定行索引划分。
2. 使用训练集调用 `fit_missingness_plan`，得到机制、缺失率、特征范围和分位阈值。
3. 使用同一个 `MissingnessPlan` 分别调用 `apply_missingness` 处理训练、验证和测试集。
4. 各模型在划分内部使用相同 mask 和同一套填充、编码规则。
5. 测试集只用于最终评价，缺失机制参数不接受测试集反馈。

`generate_missingness` 是单数据集便捷入口，会在传入数据上拟合并立即应用。需要跨划分复用时必须使用 `fit_missingness_plan` 与 `apply_missingness`，不要在验证集和测试集上分别重新拟合阈值。

## 代码接口

```python
plan = fit_missingness_plan(
    train,
    mechanism="mar",
    rate=0.3,
    seed=42,
    eligible_features=tuple(range(20)),
    driver_features=(1,),  # German Credit duration
    direction="higher",
    strength=1.0,
)

train_missing = apply_missingness(train, plan)
validation_missing = apply_missingness(validation, plan)
test_missing = apply_missingness(test, plan)
```

需要保存实验元数据时：

```python
result = apply_missingness(test, plan, return_report=True)
result.data  # {x, mask, y}
result.report.requested_rate
result.report.injected_rate
result.report.overall_missing_rate
```

`eligible_features=None` 表示所有原始字段。MAR 的 `driver_features` 必须是 `eligible_features` 的子集，并且至少保留一个可注入字段。MNAR 推荐显式传入数值或有序字段，不应对 German Credit 的全部类别枚举直接使用。

## German Credit 建议参数

| 机制 | `eligible_features` | `driver_features` | 说明 |
| --- | --- | --- | --- |
| MCAR | 全部 0 到 19 | 不适用 | 在 20 个原始字段上均匀注入 |
| MAR | 全部 0 到 19 | `[1]`，duration | duration 始终可观测，其余字段按期限高低加权 |
| MNAR | `[1, 4, 7, 10, 12, 15, 17]` | 不适用 | 仅对数值或有序字段按自身数值加权 |

不同机制使用不同可注入列集合时，实际缺失率的分母也不同。报告必须同时写出机制、候选列、目标缺失率、实际新增缺失率和整体缺失率，不能只写一个百分比。

## 测试与验收

- 输入 `x`、`mask`、`y` 不被修改。
- 输出 NaN 与 `mask=0` 完全对齐。
- 原有缺失值保持缺失。
- 标签保持逐元素相同。
- 相同数据和 seed 得到相同缺失位置。
- `0.1`、`0.3`、`0.5` 的新增缺失数精确到最近整数。
- 训练集拟合的阈值应用到其他划分时不重新计算。
- MAR 驱动列保持可观测，并表现出驱动值相关差异。
- 训练集或待应用划分的任一 MAR 驱动列已有缺失或非有限值时明确报错；目标列已有缺失仍可保留。
- MNAR 表现出自身特征值相关差异。
- 标签变化不会改变缺失位置。

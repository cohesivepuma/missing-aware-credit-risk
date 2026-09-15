# 数据获取、检查与共享协议

## 快速运行

激活 Python 3.11 虚拟环境后，在仓库根目录执行：

```bash
python scripts/prepare_german_credit.py
python experiments/run_baseline.py --config configs/german_credit.yaml
```

首次准备需要网络，后续复用经过校验的本地原始数据。准备脚本不拟合编码器、填充器或缩放器。默认 toy baseline 仍可离线执行，无需真实数据。

准备后可通过 `python scripts/inspect_german_credit.py` 仅查看五行原始样本及类型。后续查看完整数据时应通过代码生成摘要，避免打印全量记录。

## 官方来源与校验

数据为 [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)。本项目使用官方压缩包中的 **german.data 原始 20 字段格式**，保留原始字段单位用于后续缺失实验；不使用已扩展指示变量的 german.data-numeric。

引用：Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77。数据许可为 CC BY 4.0。

官方原始文件的固定 SHA256：

```text
b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871
```

下载脚本记录官方 URL、UTC 下载时间、压缩包/原始文件哈希和引用信息。下载数据不匹配固定校验值时停止；已有原始文件被修改时也停止，不自动覆盖。加载 German Credit 时还会验证处理后 CSV 的哈希、列顺序及行数。

## 本地产物

```text
data/raw/german_credit/
├── uci-german-credit.zip
├── german.data
└── download_metadata.json
data/processed/german_credit/
├── german_credit.csv         # 20 个原始字段的数值载体 + bad_credit
├── metadata.json             # 来源、校验值与质量摘要
├── schema.json               # 字段顺序、类型、类别代码映射、mask 协议
├── quality_report.json       # 缺失、重复、类别比例与数值范围
└── splits_seed42.json        # 固定划分的原始零基行索引
```

这些产物都被 Git 忽略。通过脚本和配置生成，不提交数据、权重或缓存。其他种子使用 `--seed` 准备相应的行索引文件；实验配置中的 seed 和 manifest_path 必须同步更改。

## 质量检查与标签语义

实测官方文件：1000 行、20 个特征（7 个数值字段、13 个类别字段），0 个缺失单元、0 个缺失标签、0 条完整重复记录、0 条重复特征记录。Good 为 700 条，Bad 为 300 条。数值合法性与类别枚举在转换时检查，重复数据只报告、不静默删除。

目标列为 bad_credit，**原始 1=Good 映射为 0，2=Bad 映射为 1**。标签是信用风险类别；不要把它写成具有特定违约期限的真实违约事件。目标不进入 x，也不参与特征缺失选择。

A65 等“未知/无账户”代码保留为可观测类别，不等同于 NaN；后续实验中缺失以 NaN 表示。

## 统一接口与类别处理

```python
from src.data.loader import load_dataset
from src.data.german_credit import categorical_feature_indices
from src.models.logistic import LogisticRegressionModel

data = load_dataset("german_credit")
model = LogisticRegressionModel(categorical_features=categorical_feature_indices())
```

数据结构始终为 `{x, mask, y}`，x 为 float64，mask 为 uint8，y 为 int64。**mask 为 1=observed、0=missing，且与原始 20 字段的 x 对齐。**

原始类别代码按官方枚举的固定顺序转换为 0、1、2 等数值载体；这一步不学习全数据的类别或频率，也不赋予类别连续/大小含义。训练时数值字段执行中位数填充和标准化，类别字段执行众数填充和 one-hot。所有统计量、类别集合只在训练集拟合；测试集新类别按 handle_unknown=ignore 处理，不被加入训练类别集合。填充和编码均不改变原始 mask。

`configs/german_credit.yaml` 已指定正确的类别字段索引。直接将这些载体当作普通连续变量会改变建模假设；其他模型应读取 schema 并明确自己的类别处理策略。

## 共享划分与缺失注入

```python
from src.data.preprocess import split_dataset

splits = split_dataset(
    data, seed=42, validation_size=0.2, test_size=0.2,
    calibration_fraction=0.5,
    manifest_path="data/processed/german_credit/splits_seed42.json",
)
```

manifest_path 相对于仓库根目录；存在时复用索引并验证数据指纹、配置和划分完整性，不存在时生成并保存。

| 视图 | 样本数 | Good / Bad | 用途 |
| --- | ---: | ---: | --- |
| train | 600 | 420 / 180 | 拟合预处理和基础模型 |
| validation_tune | 100 | 70 / 30 | 调参、早停、选阈值 |
| validation_calibration | 100 | 70 / 30 | 冻结模型后的概率校准 |
| test | 200 | 140 / 60 | 最终评价 |
| validation | 200 | 140 / 60 | 两个验证子集的合并视图 |

前四个视图互斥、合计 1000 行。validation 是父视图，不能再作为独立样本数相加，也不能用全部 validation 早停后又在其中一半校准。未指定 calibration_fraction 时保持旧的 train/validation/test 三视图接口。

**共享协议：先划分 → 对原始 20 个字段注入缺失 → 在训练集拟合填充/编码/缩放 → 训练与校准 → 测试。** 缺失生成器操作编码前的字段列，每个字段对应一个 mask 位；不独立掩盖 one-hot 的不同列，不掩盖 y 或行索引。机制拟合参数只能来自 train，同一比较共享掩码。后续深度模型可内部编码 x，再拼接原始 20 维 mask。

字段表和可用于课程报告的数据/预处理正文见 [docs/data_protocol.md](../docs/data_protocol.md)。缺失生成、神经网络和校准实现仍由各模块 Issue 完成。

## 其他数据入口

`load_dataset("toy", seed=42)` 生成合成数据，mask 全为 1。`load_dataset("csv", path=..., target_column=...)` 仍支持带列名的数值 CSV，目标必须是 0/1，特征可有 NaN，但不会自动推断类别字段或转换其他数据集的标签。

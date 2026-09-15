# 实验结果

执行 `python experiments/run_baseline.py` 会在本地生成 `baseline.json`，记录：

- 完整 YAML 配置（含随机种子、数据参数与评价设置）；
- 输入 x / mask / y 的 SHA256 指纹；
- train / validation / test 样本数；
- 测试集 AUC、F1、Brier Score、ECE；
- Python、平台和运行所用核心包的版本。

报告没有随机时间戳，便于同环境下直接比较重复运行的结果。指纹描述当前数值数组，用于识别相同输入，不代替原始下载文件的校验。

默认重复运行会覆盖 `baseline.json`；通过 `--output results/<run-name>.json` 保存不同实验。本目录除本说明外均被 Git 忽略，不提交缓存、权重或大量实验产物。研究报告中需要保留的汇总表和图应在后续明确约定单独的版本管理位置。

执行 `python experiments/run_calibration.py` 会在本地生成 `calibration.json` 和 `reliability_<数据名>_seed<种子>.png`，记录：

- 基础模型冻结后在 validation_calibration 上拟合的校准方法与已拟合参数；
- 校准拟合与评价的划分名、样本数、裁剪值、阈值与分箱数；
- `uncalibrated`、`platt`、`isotonic` 在同一批测试样本上的 AUC、F1、Brier、ECE；
- 各方法相对未校准的指标差值，Brier/ECE 的负值表示改善；
- 多种子运行时各变体指标的均值与标准差；
- 可靠性图路径，图内含理想线、每箱平均预测概率、正类频率与样本数。

校准可能改善也可能退化；报告只记录实际差值，不保证指标必然下降。参见 [docs/calibration_protocol.md](../docs/calibration_protocol.md)。

未来结果需包含机制、目标缺失率、实际缺失率、模型与校准方法，以及多种子均值和标准差。校准前后使用相同测试样本；mask 始终为 `1 = observed`、`0 = missing`。

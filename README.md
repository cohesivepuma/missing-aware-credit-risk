# AWARE — 缺失数据下的信用分类与概率校准

AWARE 是一个本地数据分析工作台。它帮助用户查看信用数据中的缺失字段，比较二分类模型的预测表现，并检查模型给出的概率是否可靠。评分结果用于分析和人工复核，不直接作授信决定。

仓库包含 React / TypeScript 前端、FastAPI + SQLite 后端，以及独立的 Python 研究代码。工作台支持 CSV 导入、缺失情况检查、模型训练、原始与校准概率对比、可靠性曲线、批量评分和导出。演示模式自动生成明确标注的合成样例，指标与评分均由模型真实计算。

## 快速运行工作台

环境要求 Python 3.11、Node.js 22.12+。首次安装：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-platform.txt
./scripts/run_platform.sh --demo
```

打开 <http://127.0.0.1:8000>。首次启动会构建前端、生成 1,200 行合成样例，并训练 Logistic Regression 与 Random Forest。数据集、分析报告、模型和评分批次保存在被 Git 忽略的 `var/demo/`，重启后仍可使用。

如需本地密码登录，运行 `./scripts/run_platform.sh --private`，首次设置至少 12 位密码。两种模式均默认只监听 `127.0.0.1`。使用 uv 管理环境时，可用 `uv pip install --python .venv/bin/python -r requirements-platform.txt` 安装依赖。

| 页面 | 主要操作 |
| --- | --- |
| 工作台 | 查看数据集、分析和评分记录 |
| 数据资产 | 上传 CSV，检查字段类型、缺失率与样本预览 |
| 分析实验 | 指定目标和正类含义，对比五种模型、三种概率处理方式与可靠性曲线 |
| 批量评分 | 使用冻结的模型处理新 CSV，查看输入质量并导出概率 |
| 方法说明 | 解释划分、指标、缺失处理和使用边界 |

安装细节、CSV 规则和五分钟演示流程见 [产品演示指南](docs/platform_guide.md)。接口格式见 [API 契约](docs/platform_contract.md)。当前版本面向单用户本地演示与分析；合成数据上的表现不能代表真实信贷场景。

## 研究代码与复现

项目保留 German Credit、Credit Approval、Taiwan Default 三个公开数据集的准备脚本，以及 MCAR / MAR / MNAR 缺失模拟、模型对照、独立校准和分组划分的实验代码。实验输入和输出在本地生成，不随仓库发布。研究实验采用独立协议，其指标与工作台单次 holdout 的指标不能混用。

```bash
.venv/bin/python scripts/prepare_german_credit.py
.venv/bin/python scripts/prepare_credit_benchmarks.py
.venv/bin/python experiments/run_shift_study.py
.venv/bin/python scripts/export_shift_results.py
```

完整实验占用较长时间和数百 MB 的本地结果空间；`results/shift_study/` 保存逐折输出，`results/shift_summary/` 保存汇总 CSV / JSON。运行器会检查数据、配置、代码与环境的哈希；修改实现或配置后应使用新的输出目录。macOS 上 LightGBM 可能需要 `brew install libomp`。

原有 German Credit 单样本交互演示仍可通过 `./scripts/run_demo.sh` 打开，默认地址为 <http://127.0.0.1:8501>。主要产品演示入口是上面的前后端工作台。

实验定义见 [数据协议](docs/data_protocol.md)、[缺失协议](docs/missingness_protocol.md)、[校准协议](docs/calibration_protocol.md)和[跨数据集协议](docs/shift_protocol.md)。`mask=1` 表示字段可观测，`mask=0` 表示缺失；模型输出 `[P(y=0), P(y=1)]`，正类含义需按数据集明确指定。

## 验证

```bash
.venv/bin/python -m pytest
.venv/bin/python experiments/run_baseline.py
.venv/bin/python experiments/run_calibration.py
cd web && npm ci && npm run build
```

CI 同时运行 Python 测试、重复实验一致性检查和前端构建。用户上传的数据、原始下载、运行结果、模型文件、缓存和本地工作区均不提交到 Git。

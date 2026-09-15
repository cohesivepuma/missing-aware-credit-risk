# 协作约定

每项研究任务先建立 Issue，写明目标、实验设计和验收标准，再从 `main` 建立短期分支，例如 `feat/mcar-generator`、`fix/ece-boundaries` 或 `docs/dataset-protocol`。通过 Pull Request 合并，建议至少一位同学审核；本仓库尚未配置强制分支保护。

使用 Conventional Commits，例如 `feat: add reproducible MCAR injection`。优先提交可独立检查的小改动，并同步更新配置、接口说明和必要的行为测试。

## 代码和实验规范

- Python 3.11，公开函数写 type hints，重要边界写 docstring。
- 数据契约为 `{x, mask, y}`；mask 始终为 `1 = observed`、`0 = missing`。
- 模型外部接口为 `fit(x, y, mask=None)` 和 `predict_proba(x, mask=None)`；概率列固定 `[0, 1]`。
- 指标模块返回 `auc`、`f1`、`brier`、`ece`；概率与阈值语义必须保持一致。
- 所有随机过程都显式传 seed；共享划分和掩码做公平比较。
- 预处理、机制参数和模型只在训练集拟合；校准只在验证集拟合；测试集用于最终评价。
- 不提交大型真实数据、权重、环境文件夹和缓存。依赖改动需要说明原因，并检查 Python 3.11 兼容性。
- 保持模块简单，不增加数据库、后端 API、Docker 或超出研究主线的模型。

## 提交前验证

```bash
python -m pytest
python experiments/run_baseline.py
```

PR 中写明实际运行的命令、结果与实验影响。未运行的验证要明确标注；接口 TODO 不应被描述为已经完成的研究方法。

## 建议首批 Issues

1. **接入 German Credit 数据与固定划分**：官方来源、标签映射、类别处理、数据校验及可重复转换脚本。
2. **实现并验证 MCAR 缺失生成器**：复制输入、保持标签与已缺失单元、固定 seed、目标 / 实际缺失率、mask 对齐测试。
3. **设计 MAR / MNAR 协议**：先明确可观测驱动变量、机制参数拟合范围及缺失率控制，再实现和验证。
4. **增加 Random Forest / LightGBM / MLP baseline**：统一概率接口、训练集预处理和相同划分 / 掩码。
5. **实现 Mask-aware MLP 与 mask 消融**：拼接填充值与原始 mask，并对比相同网络预算的普通 MLP。
6. **验证集概率校准与可靠性图**：先实现 Platt / Isotonic，再定义 Temperature Scaling 的 logits adapter。
7. **多种子实验汇总**：机制 × 缺失率 × 模型矩阵，记录配置、环境、实际缺失率和均值 / 标准差。

# 项目代码约定

这是 Python 3.11 的课程机器学习研究项目。需求与实现状态见 README，协作方式见 CONTRIBUTING.md。

- 先检查 Git 状态和已有代码，保留协作者的未提交改动。
- 保持第一阶段最小可运行 baseline，不把高级接口 TODO 当作已实现算法。
- 数据使用 `{x, mask, y}`；mask 只能采用 `1 = observed`、`0 = missing`。
- 外部模型调用统一 fit / predict_proba；概率列为 `[P(y=0), P(y=1)]`。
- 所有随机过程使用统一种子；预处理只在训练集拟合，校准只在验证集拟合。
- 不提交大型数据、权重或缓存；不引入数据库、API、Docker 或研究范围外技术。
- 运行 `python -m pytest` 与 `python experiments/run_baseline.py`，如未执行需明确说明。

# 项目代码约定

这是 Python 3.11 的机器学习研究与产品演示项目。需求与实现状态见 README，协作方式见 CONTRIBUTING.md。

- 先检查 Git 状态和已有代码，保留协作者的未提交改动。
- 保持第一阶段最小可运行 baseline，不把高级接口 TODO 当作已实现算法。
- 数据使用 `{x, mask, y}`；mask 只能采用 `1 = observed`、`0 = missing`。
- 外部模型调用统一 fit / predict_proba；概率列为 `[P(y=0), P(y=1)]`。
- 所有随机过程使用统一种子；预处理只在训练集拟合，校准只在验证集拟合。
- 不提交大型数据、权重、用户上传数据或缓存。
- 按用户明确授权，产品层使用 `service/` FastAPI、SQLite 与 `web/` React；研究实验入口保持独立，不修改已发表述的实验协议。
- 本地演示模式只绑定 loopback，公开部署须使用正式认证与 HTTPS；API 不接受上传模型或任意文件路径。
- 运行 `python -m pytest` 与 `python experiments/run_baseline.py`，如未执行需明确说明。

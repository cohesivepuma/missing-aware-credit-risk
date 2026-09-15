# 数据获取与生成

## 当前阶段

`load_dataset(name="toy", seed=42)` 使用 sklearn `make_classification` 在内存中生成二分类数据，不需要网络、账号或下载，不写入真实数据文件。重复使用相同参数和种子可以生成相同数据。

统一返回 `{"x": ..., "mask": ..., "y": ...}`。原始缺失值表示为 NaN，**mask 为 `1 = observed`、`0 = missing`**；toy 数据 mask 全为 1。原始 y 是合成标签，不代表真实客户违约。

## 后续公开信用数据

候选数据：[UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)。官方提供原始类别格式 `german.data` 和数值格式 `german.data-numeric`；标签为 1=Good、2=Bad，接入本项目必须映射为 0=Good、1=Bad。使用时在论文和数据说明中保留官方引用与许可信息。

后续接入流程：

1. 从官方页面下载文件到本地 `data/raw/`，记录下载地址、日期和 SHA256。
2. 按官方说明解析列、检查标签与特征类型，保持原始文件不变。
3. 映射二分类目标，移除标识符和可能泄露标签的列；明确原始缺失值。
4. 固定种子做分层划分，保留行索引。类别编码、填充、标准化和缺失机制参数只能在训练数据上拟合。
5. 将本地中间产物保存到 `data/processed/`，将转换代码与配置提交 Git。

**本阶段未实现该数据的自动下载与转换，也没有下载真实信用数据。**

## 当前本地 CSV 接口

`load_dataset(name="csv", path=..., target_column=..., seed=42)` 支持带列名的数值 CSV。特征必须全部为数值，可包含 NaN；目标必须包含 0、1 两类，不能缺失。不要把原始 1 / 2 标签未经转换直接传入。当前接口不自动解析 UCI 空格分隔文件，不自动编码类别。

复制 `configs/baseline.yaml` 为本地自定义配置，替换 `data` 部分即可：

```yaml
data:
  name: csv
  path: data/raw/credit.csv
  target_column: default
```

高级 `generate_missingness(data, mechanism=..., rate=..., seed=...)` 当前只有接口，调用会抛出 `NotImplementedError`。后续实现将复制数据而非修改输入，保持 y 不变、注入 NaN，并将对应 mask 设为 0。

`data/raw/` 和 `data/processed/` 均已加入 `.gitignore`。不要提交真实大型数据、个人敏感信息或实验缓存。

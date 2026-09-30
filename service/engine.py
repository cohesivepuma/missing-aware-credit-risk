"""Pure, reproducible training and scoring for the internal AWARE workbench.

This module deliberately keeps the research experiment runners unchanged. The
workbench fits each existing model on training rows and its probability
calibrators on a separate calibration partition. Uploaded rows never refit a
stored model or its schema.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import re
import warnings
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from src.calibration.calibrators import ProbabilityCalibrator
from src.calibration.reliability import reliability_bins
from src.metrics.metrics import evaluate_metrics, validate_probabilities

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 50_000
MAX_COLUMNS = 100
MIN_LABELED_ROWS = 100
MAX_CATEGORIES = 100
MAX_NUMERIC_MAGNITUDE = 1e12
MAX_ENCODED_CELLS = 20_000_000
MODEL_NAMES = ("logistic", "random_forest", "lightgbm", "mlp", "mask_aware_mlp")
CALIBRATIONS = ("raw", "platt", "isotonic")
MODEL_PARAMETERS = {
    "logistic": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 120, "min_samples_leaf": 3, "max_depth": 10},
    "lightgbm": {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 15,
                 "min_child_samples": 20, "reg_lambda": 1.0},
    "mlp": {"hidden_sizes": (32, 16), "epochs": 40, "batch_size": 128,
            "learning_rate": 0.001, "weight_decay": 0.001},
    "mask_aware_mlp": {"hidden_sizes": (32, 16), "epochs": 40, "batch_size": 128,
                       "learning_rate": 0.001, "weight_decay": 0.001},
}
_NUMBER = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?)$", re.I)


def _json_safe(value: Any) -> Any:
    """Only ordinary JSON scalars escape the report/summary boundary."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _clean_value(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return np.nan if text in ("", "?") else text


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("数据必须是二维表格。")
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"CSV 必须包含 1 至 {MAX_ROWS:,} 行数据，不能仅有表头。")
    if not 1 <= len(frame.columns) <= MAX_COLUMNS:
        raise ValueError(f"CSV 必须包含 1 至 {MAX_COLUMNS} 列。")
    if any(not isinstance(name, str) or not name.strip() for name in frame.columns):
        raise ValueError("CSV 列名不能为空。")
    names = [name.strip() for name in frame.columns]
    if len(set(names)) != len(names):
        raise ValueError("CSV 含有重复列名，请为每一列设置唯一名称。")
    result = frame.copy().reset_index(drop=True)
    result.columns = names
    return result.map(_clean_value)


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Parse UTF-8 CSV without pandas' automatic NA vocabulary or header repair."""
    if not isinstance(content, bytes) or not content:
        raise ValueError("CSV 文件为空。")
    if len(content) > MAX_BYTES:
        raise ValueError("CSV 文件不能超过 20 MiB。")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV 必须使用 UTF-8 编码，请重新导出后上传。") from exc
    if "\x00" in text:
        raise ValueError("CSV 含有无效的空字符，请上传文本 CSV 文件。")
    csv.field_size_limit(MAX_BYTES)
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, None)
        if header is None or not header:
            raise ValueError("CSV 文件为空或缺少表头。")
        header = [name.strip() for name in header]
        if any(not name for name in header):
            raise ValueError("CSV 列名不能为空。")
        if len(set(header)) != len(header):
            raise ValueError("CSV 含有重复列名，请为每一列设置唯一名称。")
        if len(header) > MAX_COLUMNS:
            raise ValueError(f"CSV 最多允许 {MAX_COLUMNS} 列。")
        rows: list[list[str]] = []
        for row in reader:
            if not row:  # Ignore physical blank lines, not rows of empty fields.
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV 第 {reader.line_num} 行的字段数与表头不一致。")
            rows.append(row)
            if len(rows) > MAX_ROWS:
                raise ValueError(f"CSV 最多允许 {MAX_ROWS:,} 行数据。")
    except csv.Error as exc:
        raise ValueError(f"CSV 格式无效：{exc}") from exc
    return _clean_frame(pd.DataFrame(rows, columns=header, dtype=object))


def _kind(series: pd.Series) -> str:
    observed = series.dropna()
    return "numeric" if observed.map(lambda value: bool(_NUMBER.fullmatch(value))).all() else "categorical"


def profile_frame(frame: pd.DataFrame) -> dict:
    frame = _clean_frame(frame)
    columns, notices = [], []
    for name in frame.columns:
        column = frame[name]
        missing = int(column.isna().sum())
        kind = _kind(column)
        unique = int(column.nunique(dropna=True))
        columns.append({"name": name, "kind": kind, "missing_count": missing,
                        "missing_rate": missing / len(frame), "unique_count": unique,
                        "sample_values": column.dropna().drop_duplicates().head(8).tolist()})
        if missing == len(frame):
            notices.append(f"字段「{name}」全部缺失，无法提供观测值信息。")
        elif missing / len(frame) >= 0.5:
            notices.append(f"字段「{name}」缺失率超过或等于 50%。")
        if kind == "categorical" and unique > MAX_CATEGORIES:
            notices.append(f"字段「{name}」有 {unique} 个类别；训练特征最多支持 {MAX_CATEGORIES} 个类别，可将标识字段设为 ID。")
    preview = frame.head(8).where(frame.notna(), None).to_dict(orient="records")
    return _json_safe({"rows": len(frame), "column_count": len(frame.columns),
                       "missing_rate": float(frame.isna().to_numpy().mean()),
                       "columns": columns, "preview": preview, "warnings": notices})


def demo_frame() -> pd.DataFrame:
    """Return deterministic fictitious credit data; it contains no real people."""
    rng = np.random.default_rng(20260928)
    n = 1200
    age = rng.integers(21, 71, n)
    income = np.round(np.clip(rng.lognormal(11.1, 0.52, n), 18_000, 360_000), -2)
    debt = np.round(np.clip(rng.beta(2.2, 4.2, n), 0.01, 0.92), 3)
    history = np.maximum(0, age - 20 - rng.integers(0, 15, n))
    employment = rng.choice(["salaried", "self_employed", "contract"], n, p=[0.65, 0.22, 0.13])
    past_due = rng.poisson(0.45 + debt, n)
    loan = np.round(np.clip(rng.lognormal(10.0, 0.62, n), 3_000, 130_000), -2)
    savings = np.round(rng.exponential(18_000, n), -2)
    logits = (-2.1 + 2.9 * debt + 0.62 * past_due + 0.65 * (employment == "contract")
              + 0.7 * loan / income - 0.025 * history - 0.000008 * savings)
    outcome = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    frame = pd.DataFrame({
        "age": age, "annual_income": income, "loan_amount": loan, "debt_to_income": debt,
        "credit_history_years": history,
        "employment_years": np.maximum(0, np.minimum(age - 19, rng.poisson(7, n))),
        "past_due_12m": past_due, "open_credit_lines": rng.integers(1, 13, n),
        "savings_balance": savings,
        "home_ownership": rng.choice(["rent", "mortgage", "own"], n, p=[0.36, 0.44, 0.20]),
        "employment_type": employment,
        "loan_purpose": rng.choice(["vehicle", "education", "home_improvement", "working_capital"], n),
        "outcome": outcome,
    })
    frame = _clean_frame(frame)
    for name in frame.columns.drop("outcome"):
        rate = 0.06 + (0.13 if name in ("annual_income", "savings_balance") else 0)
        probability = rate + 0.08 * (employment == "self_employed")
        frame.loc[rng.random(n) < probability, name] = np.nan
    frame.attrs["source"] = "synthetic"
    return frame


def _column_list(config: dict, key: str, *, allow_empty: bool = False) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"配置项 {key} 必须是列名列表。")
    if (not value and not allow_empty) or len(set(value)) != len(value):
        raise ValueError(f"配置项 {key} 不能为空或包含重复值。")
    return list(value)


def _numeric_values(series: pd.Series, name: str) -> np.ndarray:
    observed = series.notna().to_numpy()
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    invalid = observed & (~np.isfinite(numeric) | (np.abs(numeric) > MAX_NUMERIC_MAGNITUDE))
    if invalid.any():
        rows = ", ".join(str(int(index) + 1) for index in np.flatnonzero(invalid)[:5])
        raise ValueError(f"数值字段「{name}」存在无效数字、非有限值或绝对值超过 1e12 的值（数据行 {rows}）；请修正数据或将其配置为分类字段。")
    return numeric


def _validated(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict, np.ndarray]:
    frame = _clean_frame(frame)
    if not isinstance(config, dict):
        raise ValueError("训练配置必须是对象。")
    target = config.get("target_column")
    if not isinstance(target, str) or target not in frame:
        raise ValueError("请选择存在于数据中的目标列。")
    features = _column_list(config, "feature_columns")
    missing = [name for name in features if name not in frame]
    if missing:
        raise ValueError(f"找不到训练特征：{', '.join(missing)}。")
    if target in features:
        raise ValueError("目标列不能同时作为训练特征。")
    identifier = config.get("id_column")
    if identifier is not None:
        if not isinstance(identifier, str) or identifier not in frame:
            raise ValueError("指定的 ID 列不存在。")
        if identifier == target or identifier in features:
            raise ValueError("ID 列必须与目标列和训练特征分开。")
        _validate_ids(frame[identifier], identifier)
    if "positive_value" not in config or config["positive_value"] is None:
        raise ValueError("请明确指定目标列的正类值。")
    positive = str(config["positive_value"]).strip()
    label = config.get("positive_label")
    if not isinstance(label, str) or not label.strip() or len(label.strip()) > 120:
        raise ValueError("请填写 1 至 120 个字符的正类含义。")
    labeled = np.flatnonzero(frame[target].notna().to_numpy())
    if len(labeled) < MIN_LABELED_ROWS:
        raise ValueError(f"训练至少需要 {MIN_LABELED_ROWS} 行目标值非空的数据；当前为 {len(labeled)} 行。")
    classes = frame[target].dropna().unique().tolist()
    if len(classes) != 2:
        raise ValueError("目标列必须恰好包含两个非空类别。")
    if positive not in classes:
        raise ValueError("指定的正类值不在目标列的两个类别中。")
    categorical = (_column_list(config, "categorical_columns", allow_empty=True)
                   if config.get("categorical_columns") is not None
                   else [name for name in features if _kind(frame[name]) == "categorical"])
    if any(name not in features for name in categorical):
        raise ValueError("分类字段必须包含在已选训练特征中。")
    width = 0
    for name in features:
        if name in categorical:
            count = frame[name].nunique(dropna=True)
            if count > MAX_CATEGORIES:
                raise ValueError(f"分类特征「{name}」有 {count} 个类别，超过 {MAX_CATEGORIES} 个的上限；标识字段应设为 ID 或排除。")
            width += max(1, count)
        else:
            _numeric_values(frame[name], name)
            width += 1
    models = _column_list(config, "models")
    if any(name not in MODEL_NAMES for name in models):
        raise ValueError(f"支持的模型为：{', '.join(MODEL_NAMES)}。")
    if any(model != "lightgbm" for model in models) and width * len(labeled) > MAX_ENCODED_CELLS:
        raise ValueError("分类特征展开后的数据过大；请减少类别数、特征数或训练行数。")
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("随机种子必须是 0 至 4294967295 之间的整数。")
    normalized = dict(config, feature_columns=features, categorical_columns=categorical,
                      target_column=target, id_column=identifier, positive_value=positive,
                      positive_label=label.strip(), models=models, seed=seed)
    return frame, normalized, labeled


def _validate_ids(series: pd.Series, name: str) -> None:
    if series.isna().any() or series.duplicated().any():
        raise ValueError(f"ID 字段「{name}」必须在整个文件中非空且唯一。")


def _feature_hashes(frame: pd.DataFrame, config: dict) -> list[str]:
    canonical = frame[config["feature_columns"]].copy()
    for name in config["feature_columns"]:
        if name not in config["categorical_columns"]:
            values = _numeric_values(canonical[name], name)
            values[values == 0] = 0.0  # Normalize -0 and alternative numeric spellings.
            canonical[name] = values
    hashes = []
    for row in canonical.itertuples(index=False, name=None):
        values = [None if pd.isna(value) else value for value in row]
        payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        hashes.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    return hashes


def _partition(frame: pd.DataFrame, config: dict, labeled: np.ndarray) -> tuple[dict, list[str]]:
    """Group on feature values alone, then stratify five approximately equal folds.

    Candidate assignments are judged solely by sizes and class counts. Model
    scores are never involved. Numeric-equivalent CSV strings share a group.
    """
    hashes = _feature_hashes(frame.iloc[labeled], config)
    groups = np.asarray(hashes)
    y = (frame.iloc[labeled][config["target_column"]] == config["positive_value"]).to_numpy(dtype=int)
    for label in (0, 1):
        if len(np.unique(groups[y == label])) < 5:
            raise ValueError("每个目标类别至少需要 5 组独立的特征记录；相同特征的重复行必须放在同一分区。")
    best, best_loss = None, float("inf")
    targets = np.asarray([0.6, 0.2, 0.2])
    totals = np.bincount(y, minlength=2)
    for attempt in range(8):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=(config["seed"] + attempt) % 2**32)
        folds = [rows for _, rows in splitter.split(np.zeros(len(y)), y, groups)]
        for cal, test in combinations(range(5), 2):
            parts = [np.concatenate([folds[i] for i in range(5) if i not in (cal, test)]), folds[cal], folds[test]]
            counts = np.asarray([np.bincount(y[rows], minlength=2) for rows in parts])
            if (counts == 0).any():
                continue
            loss = float(np.square(counts / totals - targets[:, None]).sum())
            if loss < best_loss:
                best_loss, best = loss, parts
        if best_loss < 0.0002:
            break
    if best is None:
        raise ValueError("无法在保持重复特征组独立的同时，使训练、校准和测试分区都包含两个类别；请增加独立样本。")
    sizes = np.asarray([len(rows) / len(y) for rows in best])
    if ((sizes < np.array([0.4, 0.1, 0.1])) | (sizes > np.array([0.8, 0.35, 0.35]))).any():
        raise ValueError("重复特征组过大，无法形成约 60% / 20% / 20% 的独立分区；请增加独立样本。")
    return {name: np.sort(labeled[rows]) for name, rows in zip(("train", "calibration", "test"), best)}, hashes


def validate_training(frame: pd.DataFrame, config: dict) -> None:
    frame, config, labeled = _validated(frame, config)
    _partition(frame, config, labeled)


def _encode(frame: pd.DataFrame, schema: dict) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    values, missing, unseen_rates = [], [], {}
    for name in schema["feature_columns"]:
        series = frame[name]
        observed = series.notna().to_numpy()
        if name in schema["categorical_columns"]:
            mapping = schema["category_maps"][name]
            # -1 is finite and cannot be a train category. Missing alone is NaN.
            encoded = series.map(mapping).fillna(-1).to_numpy(dtype=np.float64)
            encoded[~observed] = np.nan
            unseen_rates[name] = float((observed & (encoded == -1)).mean())
        else:
            encoded = _numeric_values(series, name)
            unseen_rates[name] = 0.0
        values.append(encoded)
        missing.append(observed)
    return np.column_stack(values), np.column_stack(missing), unseen_rates


def _make_model(name: str, seed: int, categorical: list[int]):
    if name == "logistic":
        from src.models.logistic import LogisticRegressionModel
        cls = LogisticRegressionModel
    elif name == "random_forest":
        from src.models.random_forest import RandomForestModel
        cls = RandomForestModel
    elif name == "lightgbm":
        from src.models.lightgbm_model import LightGBMModel
        cls = LightGBMModel
    elif name == "mlp":
        from src.models.mlp import MLPModel
        cls = MLPModel
    else:
        from src.models.mask_aware_mlp import MaskAwareMLP
        cls = MaskAwareMLP
    try:
        return cls(seed=seed, categorical_features=categorical, **MODEL_PARAMETERS[name])
    except ImportError as exc:
        raise ValueError(f"模型 {name} 的运行依赖尚未安装：{exc.name}。") from exc


def _fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_json(orient="split", index=False, force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _package_versions() -> dict:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "lightgbm"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def train_analysis(frame: pd.DataFrame, config: dict,
                   progress: Callable[[int, str], None] | None = None) -> tuple[dict, object]:
    def emit(percent: int, message: str) -> None:
        if progress is not None:
            progress(percent, message)

    emit(3, "正在校验字段与目标类别")
    frame, config, labeled = _validated(frame, config)
    splits, hashes = _partition(frame, config, labeled)
    emit(10, "已生成独立的训练、校准和测试分区")
    train = frame.iloc[splits["train"]]
    schema = {key: config[key] for key in ("feature_columns", "categorical_columns", "id_column",
                                          "target_column", "positive_value", "positive_label")}
    schema["category_maps"] = {
        name: {value: position for position, value in enumerate(sorted(train[name].dropna().unique()))}
        for name in config["categorical_columns"]
    }
    schema["training_missing_rates"] = {name: float(train[name].isna().mean()) for name in config["feature_columns"]}
    matrices = {name: _encode(frame.iloc[rows], schema) for name, rows in splits.items()}
    labels = {name: (frame.iloc[rows][config["target_column"]] == config["positive_value"]).to_numpy(dtype=int)
              for name, rows in splits.items()}
    notices = []
    omitted = len(frame) - len(labeled)
    if omitted:
        notices.append(f"已排除 {omitted} 行目标值缺失的记录；其余 {len(labeled)} 行用于分析。")
    duplicates = len(hashes) - len(set(hashes))
    if duplicates:
        notices.append(f"检测到 {duplicates} 行重复特征记录；相同特征始终留在同一分区以防止泄漏。")
    for name in config["feature_columns"]:
        rate = schema["training_missing_rates"][name]
        if rate == 1:
            notices.append(f"字段「{name}」在训练集中全部缺失，预测时需谨慎解释。")
        elif rate >= 0.5:
            notices.append(f"字段「{name}」在训练集中的缺失率超过或等于 50%。")
        for split in ("calibration", "test"):
            unseen = matrices[split][2][name]
            if unseen:
                notices.append(f"{split} 分区的字段「{name}」含 {unseen:.1%} 训练中未见的类别，已使用冻结规则处理。")
    categorical = [config["feature_columns"].index(name) for name in config["categorical_columns"]]
    fitted, metrics, calibrator_parameters = {}, [], {}
    count = len(config["models"])
    for index, name in enumerate(config["models"]):
        emit(15 + int(75 * index / count), f"正在训练 {name}")
        estimator = _make_model(name, config["seed"], categorical)
        x_train, m_train, _ = matrices["train"]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            estimator.fit(x_train, labels["train"], mask=m_train)
        for notice in captured:
            if "converg" in str(notice.message).lower():
                notices.append(f"模型 {name} 达到固定训练预算；请查看收敛情况并谨慎解释结果。")
                break
        x_cal, m_cal, _ = matrices["calibration"]
        x_test, m_test, _ = matrices["test"]
        cal_raw = validate_probabilities(estimator.predict_proba(x_cal, mask=m_cal)[:, 1])
        test_raw = validate_probabilities(estimator.predict_proba(x_test, mask=m_test)[:, 1])
        calibrators = {method: ProbabilityCalibrator(method).fit(cal_raw, labels["calibration"])
                       for method in CALIBRATIONS if method != "raw"}
        fitted[name] = {"estimator": estimator, "calibrators": calibrators}
        calibrator_parameters[name] = {method: obj.fitted_parameters for method, obj in calibrators.items()}
        for method in CALIBRATIONS:
            probabilities = test_raw if method == "raw" else calibrators[method].predict_proba(test_raw)
            bins = reliability_bins(labels["test"], probabilities)
            metrics.append({"model": name, "calibration": method,
                            **evaluate_metrics(labels["test"], probabilities),
                            "log_loss": float(log_loss(labels["test"], probabilities, labels=[0, 1])),
                            "reliability": {key: bins[key] for key in ("counts", "mean_probability", "positive_frequency")}})
    emit(94, "正在生成指标、可靠性曲线与可复现记录")
    provenance = {
        "engine_version": 1, "seed": config["seed"], "input_rows": len(frame),
        "analyzed_rows": len(labeled), "omitted_target_rows": omitted,
        "omitted_row_indices": np.flatnonzero(frame[config["target_column"]].isna().to_numpy()),
        "dataset_sha256": _fingerprint(frame),
        "feature_sha256": hashlib.sha256("".join(hashes).encode("ascii")).hexdigest(),
        "split_method": "stratified feature-group holdout; approximately 60/20/20",
        "row_index_convention": "zero-based original CSV data row, excluding header",
        "split_indices": splits,
        "split_sha256": {name: hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()
                         for name, rows in splits.items()},
        "split_feature_sha256": {name: hashlib.sha256("".join(_feature_hashes(frame.iloc[rows], config)).encode("ascii")).hexdigest()
                                 for name, rows in splits.items()},
        "independent_feature_groups": len(set(hashes)),
        "label_mapping": {"0": next(value for value in frame[config["target_column"]].dropna().unique()
                                    if value != config["positive_value"]), "1": config["positive_value"]},
        "feature_columns": config["feature_columns"], "categorical_columns": config["categorical_columns"],
        "category_counts_train": {name: len(mapping) for name, mapping in schema["category_maps"].items()},
        "unknown_category_code": -1, "mask_convention": "1=observed, 0=missing before imputation",
        "preprocessing_fit_partition": "train", "calibrator_fit_partition": "calibration",
        "evaluation_partition": "test", "model_selection_on_test": False,
        "f1_threshold": 0.5, "reliability_bins": 10,
        "model_parameters": {name: dict(MODEL_PARAMETERS[name], seed=config["seed"], categorical_features=categorical)
                             for name in config["models"]},
        "calibrator_parameters": calibrator_parameters, "packages": _package_versions(),
    }
    result = _json_safe({
        "rows": len(labeled), "positive_label": config["positive_label"],
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "class_counts": {name: np.bincount(values, minlength=2) for name, values in labels.items()},
        "metrics": metrics,
        "feature_quality": [{"name": name, "train_missing_rate": schema["training_missing_rates"][name],
                             "test_missing_rate": float(frame.iloc[splits["test"]][name].isna().mean())}
                            for name in config["feature_columns"]],
        "warnings": notices, "provenance": provenance,
    })
    bundle = dict(schema, schema_version=1, models=fitted, provenance=result["provenance"])
    emit(100, "分析完成")
    return result, bundle


def score_batch(bundle: object, frame: pd.DataFrame, model: str, calibration: str) -> tuple[dict, pd.DataFrame]:
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("训练模型版本不兼容，请重新运行分析。")
    if model not in bundle["models"]:
        raise ValueError("所选模型不在此分析已训练的模型中。")
    if calibration not in CALIBRATIONS:
        raise ValueError("校准方法必须为 raw、platt 或 isotonic。")
    frame = _clean_frame(frame)
    required = list(bundle["feature_columns"])
    if bundle["id_column"] is not None:
        required.append(bundle["id_column"])
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"预测 CSV 缺少必需字段：{', '.join(missing)}。请使用此分析的预测模板。")
    identifier = bundle["id_column"]
    if identifier is not None:
        _validate_ids(frame[identifier], identifier)
        ids = frame[identifier].tolist()
    else:
        ids = [str(index) for index in range(1, len(frame) + 1)]
    notices = []
    extra = [name for name in frame.columns if name not in required]
    if extra:
        notices.append(f"已忽略额外字段：{', '.join(extra)}；预测仅使用训练时选定的特征。")
    values, observed, unseen = _encode(frame, bundle)
    fitted = bundle["models"][model]
    raw = validate_probabilities(fitted["estimator"].predict_proba(values, mask=observed)[:, 1])
    probability = raw if calibration == "raw" else fitted["calibrators"][calibration].predict_proba(raw)
    probability = validate_probabilities(probability)
    missing_counts = (~observed).sum(axis=1)
    quality = []
    for index, name in enumerate(bundle["feature_columns"]):
        training_rate = bundle["training_missing_rates"][name]
        rate = float((~observed[:, index]).mean())
        quality.append({"name": name, "training_missing_rate": training_rate,
                        "batch_missing_rate": rate, "unseen_rate": unseen[name]})
        if rate == 1:
            notices.append(f"字段「{name}」在本批次全部缺失。")
        elif rate >= 0.5:
            notices.append(f"字段「{name}」在本批次缺失率超过或等于 50%。")
        if rate - training_rate >= 0.15:
            notices.append(f"字段「{name}」的缺失率比训练集上升 {(rate - training_rate):.1%}，可能影响预测质量。")
        if unseen[name]:
            notices.append(f"字段「{name}」有 {unseen[name]:.1%} 的记录使用训练中未见的类别；已按冻结规则处理，模型未重新拟合。")
    all_missing = int((missing_counts == len(bundle["feature_columns"])).sum())
    if all_missing:
        notices.append(f"本批次有 {all_missing} 行全部特征缺失，其概率仅供参考。")
    scored = pd.DataFrame({"record_id": ids, "probability": probability, "missing_count": missing_counts})
    summary = _json_safe({"model": model, "calibration": calibration, "rows": len(frame),
                          "mean_probability": float(probability.mean()),
                          "missing_rate": float((~observed).mean()),
                          "positive_label": bundle["positive_label"],
                          "preview": scored.head(10).to_dict(orient="records"),
                          "quality": quality, "warnings": notices})
    return summary, scored

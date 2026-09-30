"""Interactive credit risk demonstrator backed by measured experiment code."""

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.calibration.reliability import reliability_bins
from src.data.german_credit import FEATURES
from src.research import MODEL_LABELS, fit_scenario, load_config

LABELS = ["支票账户状态", "贷款期限（月）", "信用历史", "贷款用途", "贷款金额（DM）", "储蓄账户状态",
          "就业年限", "分期付款比例（%）", "个人状态与性别", "其他债务人／担保人", "现住址年限",
          "财产类型", "年龄（岁）", "其他分期计划", "住房情况", "现有信贷数量", "工作类别", "赡养人数",
          "电话登记", "外籍劳动者"]
VARIANTS = {"uncalibrated": "校准前", "platt": "Platt 校准", "isotonic": "Isotonic 校准"}
COLORS = ["#087f75", "#4263eb", "#e6a23c", "#b05bbf"]


@st.cache_resource(show_spinner=False, max_entries=12)
def get_scenario(config_text: str, data_sha: str, metadata_sha: str, seed: int, mechanism: str, rate: float):
    """Content keys invalidate the cache when protocol or checked data changes."""
    return fit_scenario(json.loads(config_text), seed, mechanism, rate)


def metric_table(scenario) -> pd.DataFrame:
    frame = pd.DataFrame(scenario.report["records"])
    return frame[["model", "calibration", "auc", "f1", "brier", "ece"]]


def show_metrics(frame: pd.DataFrame) -> None:
    view = frame.copy()
    view["model"] = view["model"].map(MODEL_LABELS)
    view["calibration"] = view["calibration"].map(VARIANTS)
    st.dataframe(view.rename(columns={"model": "模型", "calibration": "概率处理", "auc": "AUC ↑", "f1": "F1 ↑", "brier": "Brier ↓", "ece": "ECE ↓"}).style.format({"AUC ↑": "{:.4f}", "F1 ↑": "{:.4f}", "Brier ↓": "{:.4f}", "ECE ↓": "{:.4f}"}),
                 hide_index=True, use_container_width=True)


def reliability_chart(scenario, model: str) -> None:
    records = []
    for method, p in scenario.probabilities[model].items():
        bins = reliability_bins(scenario.splits["test"]["y"], p, n_bins=10)
        for i, count in enumerate(bins["counts"]):
            if count:
                records.append({"预测概率": bins["mean_probability"][i], "实际 Bad 比例": bins["positive_frequency"][i], "样本数": int(count), "曲线": VARIANTS[method]})
    curve = alt.Chart(pd.DataFrame(records)).mark_line(point=True).encode(
        x=alt.X("预测概率:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("实际 Bad 比例:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("曲线:N", scale=alt.Scale(range=COLORS)),
        tooltip=["曲线", alt.Tooltip("预测概率", format=".3f"), alt.Tooltip("实际 Bad 比例", format=".3f"), "样本数"],
    )
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[5, 5], color="#a6afbd").encode(x="x:Q", y="y:Q")
    st.altair_chart((curve + diagonal).properties(height=310), use_container_width=True)
    st.caption("10 个等宽概率区间；空区间不画点。靠近虚线表示该区间的预测概率更接近实际频率，小样本区间波动更大。")


def overview(scenario, config) -> None:
    st.markdown('<div class="hero"><div class="eyebrow">AWARE / CREDIT RESEARCH LAB</div><h1>数据不完整，预测还能可靠吗？</h1><p>从缺失模式，到信用分类，再到概率校准。用真实公开数据观察模型的能力与边界。</p></div>', unsafe_allow_html=True)
    st.caption("本页使用 German Credit 数据演示一个缺失场景；左侧可以切换机制、比例和随机划分。")
    columns = st.columns(4)
    for col, title, value, detail in zip(columns, ["公开样本", "原始字段", "对照模型", "独立测试样本"], ["1,000", "20", "4", "200"], ["UCI German Credit", "7 数值 / 13 类别", "相同划分与缺失掩码", "仅用于最终评价"]):
        with col:
            st.metric(title, value)
            st.caption(detail)
    st.subheader("一条可追溯的研究流程")
    stages = st.columns(4)
    for col, label, body in zip(stages, ["01  构造缺失", "02  训练模型", "03  校准概率", "04  配对评价"],
                               ["MCAR / MAR / MNAR，共享六个缺失目标字段。", "600 个训练样本；比较线性、树和神经网络模型。", "100 个独立校准样本；保留 100 个调参样本。", "AUC、F1、Brier 与 ECE，保留改善与退化结果。"]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.write(body)
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("当前场景的模型表现")
        show_metrics(metric_table(scenario).query("calibration == 'uncalibrated'"))
        st.caption("左侧控制当前种子、缺失机制与比例。这里是单次测试结果，多次划分结果见“缺失实验室”。")
    with right:
        st.subheader("研究问题")
        st.markdown("**缺失信息是否有用？**\n\n比较常量掩码对照与真实掩码输入，保持网络结构和训练预算一致。\n\n**校准是否稳定有效？**\n\n观察 Platt 与 Isotonic 的实际变化，区分排序能力和概率可靠性。")
    st.info("标签为历史数据中的 Good / Bad 信用类别。页面展示 P(Bad)，不能解释为某一还款期限的违约概率或用于实际授信。")


def prediction(scenario, model: str, variant: str) -> None:
    st.title("交互预测")
    st.write("选择一个保留测试样本，调整字段并主动隐藏信息，观察同一模型的概率变化。")
    index = st.selectbox("测试样本编号", list(range(len(scenario.clean_splits["test"]["y"]))), format_func=lambda i: f"样本 {i + 1:03d}")
    base = scenario.clean_splits["test"]["x"][index].copy()
    with st.container(border=True):
        st.markdown("**编辑样本字段**")
        columns = st.columns(3)
        for i in (1, 4, 12):
            with columns[(1, 4, 12).index(i)]:
                base[i] = st.number_input(LABELS[i], min_value=1, max_value=FEATURES[i].maximum or 100000, value=int(base[i]), step=1, key=f"field_{index}_{i}")
        with st.expander("其他 17 个字段（类别选项使用 UCI 原始编码）"):
            cols = st.columns(3)
            for j, i in enumerate(i for i in range(20) if i not in (1, 4, 12)):
                feature = FEATURES[i]
                with cols[j % 3]:
                    if feature.kind == "categorical":
                        base[i] = st.selectbox(LABELS[i], list(range(len(feature.codes))), index=int(base[i]), format_func=lambda v, codes=feature.codes: codes[v], key=f"field_{index}_{i}")
                    else:
                        base[i] = st.number_input(LABELS[i], min_value=1, max_value=feature.maximum or 100000, value=int(base[i]), step=1, key=f"field_{index}_{i}")
    missing = st.multiselect("隐藏哪些字段", list(range(20)), default=[4, 12], format_func=lambda i: LABELS[i], key=f"missing_{index}")
    values = base.reshape(1, -1).copy()
    values[:, missing] = np.nan
    mask = (~np.isnan(values)).astype(np.uint8)
    estimator = scenario.models[model]
    complete = float(estimator.predict_proba(base.reshape(1, -1), np.ones((1, 20), dtype=np.uint8))[0, 1])
    raw = float(estimator.predict_proba(values, mask)[0, 1])
    calibrated = raw if variant == "uncalibrated" else float(scenario.calibrators[model][variant].predict_proba(np.array([raw]))[0])
    a, b, c, d = st.columns(4)
    a.metric("输入完整度", f"{int(mask.sum())} / 20")
    b.metric("完整输入 P(Bad)", f"{complete:.1%}")
    c.metric("缺失后 P(Bad)", f"{raw:.1%}", f"{(raw-complete)*100:+.1f} 个百分点", delta_color="off")
    d.metric("所选校准后 P(Bad)", f"{calibrated:.1%}", f"{(calibrated-raw)*100:+.1f} 个百分点", delta_color="off")
    st.caption("完整输入与缺失输入使用同一个已冻结模型；概率之差是输入扰动的响应，不是因果效应。")
    st.progress(calibrated, text=f"{MODEL_LABELS[model]} · {VARIANTS[variant]} · P(Bad) = {calibrated:.3f}")
    if not mask.any():
        st.warning("所有字段均已隐藏：该输入超出本研究训练的缺失范围，输出仅用于观察模型行为。")
    elif any(i not in (4, 7, 10, 12, 15, 17) for i in missing):
        st.info("你隐藏了预设范围之外的字段；此结果仅用于观察模型行为。")
    if st.checkbox("查看原始样本标签"):
        label = scenario.clean_splits["test"]["y"][index]
        st.write(f"原始样本：{'Bad' if label else 'Good'}。编辑后的假设样本没有已知真实标签。")
    with st.expander("当前模型在整批测试样本上的可靠性"):
        reliability_chart(scenario, model)
    st.download_button("下载本次预测记录", json.dumps({"model": model, "calibration": variant, "test_sample_position": index,
                        "scenario": {k: scenario.report[k] for k in ("seed", "mechanism", "rate", "dataset_sha256")},
                        "input": {f.name: None if i in missing else float(base[i]) for i, f in enumerate(FEATURES)},
                        "complete_probability": complete, "raw_probability": raw, "calibrated_probability": calibrated}, indent=2, ensure_ascii=False),
                       "prediction.json", "application/json")


def laboratory(scenario, config, model: str) -> None:
    st.title("缺失实验室")
    st.write("先观察当前场景，再查看跨随机划分的实验结果。每个场景内，所有模型共享样本和缺失掩码。")
    a, b = st.columns([1, 1.2])
    with a:
        st.subheader("缺失分布")
        missing = (1 - scenario.splits["test"]["mask"]).mean(axis=0)
        frame = pd.DataFrame({"字段": LABELS, "缺失比例": missing})
        chart = alt.Chart(frame[frame["缺失比例"] > 0]).mark_bar(color=COLORS[0]).encode(
            x=alt.X("缺失比例:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])), y=alt.Y("字段:N", sort="-x"), tooltip=["字段", alt.Tooltip("缺失比例", format=".1%")])
        st.altair_chart(chart.properties(height=290), use_container_width=True)
        st.caption(f"全表实际缺失率：{float(np.mean(scenario.splits['test']['mask']==0)):.1%}。目标比例只针对六个字段，不能与全表比例混用。")
    with b:
        st.subheader("校准可靠性")
        reliability_chart(scenario, model)
    show_metrics(metric_table(scenario))
    st.subheader("跨划分实验")
    path = PROJECT_ROOT / "results/benchmark.json"
    if not path.exists():
        st.info("多次划分实验尚未运行。执行下方命令生成结果。")
        st.code("python experiments/run_missingness.py")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if report.get("config_sha256") != config_hash or any(r["dataset_sha256"] != scenario.report["dataset_sha256"] for r in report["runs"]):
        st.warning("保存的实验与当前配置或数据不一致，请重新运行实验后比较。")
        return
    summary = pd.DataFrame(report["summary"])
    metric = st.selectbox("查看指标", ["auc", "brier", "ece", "f1"], format_func=lambda x: x.upper())
    subset = summary[(summary.mechanism != "complete") & (summary.calibration == "platt")].copy()
    subset["模型"] = subset.model.map(MODEL_LABELS)
    chart = alt.Chart(subset).mark_line(point=True).encode(
        x=alt.X("rate:Q", title="六个目标字段内的缺失率", axis=alt.Axis(format="%")),
        y=alt.Y(f"{metric}_mean:Q", title=f"{metric.upper()} 均值", scale=alt.Scale(zero=False)),
        color=alt.Color("模型:N", scale=alt.Scale(range=COLORS)),
        column=alt.Column("mechanism:N", title=None),
        tooltip=["模型", "mechanism", "rate", f"{metric}_mean", f"{metric}_std", "n_seeds"],
    ).properties(height=240, width=220)
    st.altair_chart(chart, use_container_width=False)
    st.caption(f"所有曲线使用 Platt 校准；{len(config['seeds'])} 次分层随机划分，悬停查看样本标准差。重复划分存在样本重叠，标准差不是置信区间。")
    st.download_button("下载完整实验摘要 CSV", summary.to_csv(index=False).encode("utf-8-sig"), "benchmark_summary.csv", "text/csv")


def main() -> None:
    st.set_page_config(page_title="AWARE · 信用风险演示", page_icon="◈", layout="wide")
    st.markdown("""<style>
    .block-container{padding-top:2rem;max-width:1280px}
    .hero{background:linear-gradient(120deg,#122a3a,#11675f);padding:36px 40px;border-radius:16px;margin-bottom:28px;color:white}
    .hero h1{color:white;font-size:2.35rem;padding:12px 0}.hero p{color:#d7ede9;font-size:1.05rem;margin:0}
    .eyebrow{letter-spacing:3px;font-size:.75rem;color:#9fddd0}
    [data-testid="stMetric"]{background:white;border:1px solid #e1e8ed;border-radius:12px;padding:16px}
    [data-testid="stSidebar"]{border-right:1px solid #e1e8ed}
    </style>""", unsafe_allow_html=True)
    config = load_config()
    with st.sidebar:
        st.markdown("## ◈ AWARE")
        st.caption("MISSING DATA · RELIABLE PROBABILITIES")
        page = st.radio("导航", ["研究总览", "交互预测", "缺失实验室"], label_visibility="collapsed")
        st.divider()
        st.markdown("**实验场景**")
        mechanism = st.selectbox("缺失机制", ["mcar", "mar", "mnar"], format_func=str.upper)
        rate = st.select_slider("目标字段缺失比例", [0.0, .1, .3, .5], value=.3, format_func=lambda v: f"{v:.0%}")
        seed = st.selectbox("随机划分种子", config["seeds"])
        model = st.selectbox("观察模型", config["models"], index=3, format_func=lambda v: MODEL_LABELS[v])
        variant = st.selectbox("概率处理", list(VARIANTS), index=1, format_func=lambda v: VARIANTS[v])
        st.caption({"mcar": "MCAR：等概率选择缺失单元。", "mar": "MAR：较长贷款期限提高目标字段的缺失倾向。", "mnar": "MNAR：字段自身值较大时，更容易缺失。"}[mechanism])
        st.caption("缺失比例作用于 6 / 20 个字段。模型会在所选场景的训练集上重新训练并缓存。")
    path = PROJECT_ROOT / config["data"]["path"]
    try:
        data_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata_sha = hashlib.sha256(path.with_name("metadata.json").read_bytes()).hexdigest()
        with st.spinner("正在准备共享划分、训练模型并拟合独立校准器…"):
            scenario = get_scenario(json.dumps(config, sort_keys=True), data_sha, metadata_sha, seed, mechanism if rate else "complete", rate)
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.code("python scripts/prepare_german_credit.py")
        st.stop()
    if page == "研究总览":
        overview(scenario, config)
    elif page == "交互预测":
        prediction(scenario, model, variant)
    else:
        laboratory(scenario, config, model)
    st.divider()
    st.caption("AWARE  ·  Interactive demo  ·  UCI German Credit  ·  Good / Bad classification")


if __name__ == "__main__":
    main()

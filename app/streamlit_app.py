"""Optional phase-later Streamlit landing page; prediction is not implemented."""

import streamlit as st


def main() -> None:
    """Present project status without pretending a credit model is deployed."""
    st.title("Missing-aware Credit Risk")
    st.write("面向不完整数据的缺失机制感知与概率校准个人信用风险评估方法")
    st.info("当前已实现 toy 数据 Logistic Regression baseline；信用预测 Demo 尚未实现。")
    st.code("python experiments/run_baseline.py", language="bash")
    st.caption("Missing mask convention: 1 = observed, 0 = missing")
    # TODO: load a verified model, accept feature/mask inputs, and show calibration.


if __name__ == "__main__":
    main()

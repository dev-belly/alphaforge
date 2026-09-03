"""AlphaForge research dashboard (Streamlit).

A click-and-run front end over :mod:`alphaforge.pipeline`. Every number shown
is produced by the same engine the CLI and the API use, so the dashboard and
the report can never disagree.

Run::

    streamlit run apps/dashboard/streamlit_app.py

(needs the editable install: ``pip install -e .[dashboard]``, or set
 ``PYTHONPATH=src`` when launching).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from alphaforge.pipeline import ResearchPipeline
from alphaforge.utils.config import Config, set_global_seed

st.set_page_config(
    page_title="AlphaForge Dashboard", layout="wide", initial_sidebar_state="expanded"
)

MODELS = ["ridge", "elasticnet", "random_forest", "lightgbm"]
METHODS = ["mean_variance", "equal_weight", "min_variance", "max_sharpe", "risk_parity"]


@st.cache_data(show_spinner=False)
def _monthly_heatmap(returns):
    from alphaforge.backtest.metrics import monthly_returns

    try:
        m = monthly_returns(returns)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if m.empty:
        return pd.DataFrame()
    return m


def _metric_card(label: str, value) -> None:
    st.metric(label, value)


def main() -> None:
    st.title("AlphaForge · Quant Research Dashboard")
    st.caption(
        "Institutional quant research & portfolio engineering — every figure comes from the live engine."
    )

    with st.sidebar:
        st.header("Configuration")
        start = st.date_input("Window start", value=date(2018, 1, 1))
        end = st.date_input("Window end", value=date(2024, 12, 31))
        model = st.selectbox("ML model", MODELS, index=0)
        method = st.selectbox("Portfolio method", METHODS, index=0)
        vol = st.slider("Target volatility (annual)", 0.05, 0.25, 0.12, 0.01)
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
        run = st.button("Run research pipeline", type="primary", use_container_width=True)
        st.divider()
        st.caption("Heavy stages (factor eval, walk-forward) run once per click.")

    if not run and "state" not in st.session_state:
        st.info(
            "Set the parameters and press **Run research pipeline** to compute a full strategy."
        )
        return

    if run or "state" not in st.session_state:
        set_global_seed(int(seed))
        overrides = {"portfolio": {"method": method, "target_volatility": float(vol)}}
        cfg = Config.load(overrides=overrides)
        with st.spinner(
            "Running data → factors → ML → risk → portfolio → backtest → attribution …"
        ):
            state = ResearchPipeline(cfg).run(
                start=str(start), end=str(end), model_type=model, report_dir="research/reports"
            )
        st.session_state["state"] = state
        st.session_state["params"] = {
            "start": str(start),
            "end": str(end),
            "model": model,
            "method": method,
            "vol": vol,
        }

    state = st.session_state["state"]
    params = st.session_state.get("params", {})
    bt = state.backtest

    if bt is None:
        st.error("Pipeline produced no backtest result.")
        return

    # ---- headline metrics --------------------------------------------------
    s = bt.summary()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        _metric_card("CAGR", f"{s.get('cagr', float('nan')):+.2%}")
    with col2:
        _metric_card("Sharpe", f"{s.get('sharpe', float('nan')):.2f}")
    with col3:
        _metric_card("Max Drawdown", f"{s.get('max_drawdown', float('nan')):.2%}")
    with col4:
        _metric_card("Volatility", f"{s.get('ann_vol', float('nan')):.2%}")
    with col5:
        _metric_card("Information Ratio", f"{s.get('information_ratio', float('nan')):.2f}")

    st.caption(
        f"Run params: model={params.get('model')} · method={params.get('method')} · "
        f"target vol={params.get('vol')} · window {params.get('start')} → {params.get('end')}"
    )

    # ---- equity curve ------------------------------------------------------
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        eq = bt.equity.dropna()
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq.values, name="Strategy", line=dict(color="#1f4e79"))
        )
        if bt.benchmark is not None:
            bm = bt.benchmark.reindex(eq.index).dropna()
            if len(bm):
                bm_eq = (1 + bm).cumprod()
                fig.add_trace(
                    go.Scatter(
                        x=bm_eq.index,
                        y=bm_eq.values,
                        name="Benchmark",
                        line=dict(color="#999", dash="dot"),
                    )
                )
        fig.update_layout(
            height=340,
            margin=dict(l=30, r=20, t=30, b=30),
            title="Equity Curve",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Equity chart unavailable: {exc}")

    # ---- lower grid: monthly heatmap + drawdown ---------------------------
    c1, c2 = st.columns(2)
    with c1:
        m = _monthly_heatmap(bt.returns)
        if not m.empty:
            try:
                import plotly.graph_objects as go

                z = m.values
                fig = go.Figure(
                    data=go.Heatmap(
                        z=z,
                        x=[str(c) for c in m.columns],
                        y=[str(i) for i in m.index],
                        colorscale="RdYlGn",
                        zmid=0,
                        colorbar=dict(title="return"),
                    )
                )
                fig.update_layout(
                    height=320, margin=dict(l=30, r=20, t=30, b=30), title="Monthly Returns"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:  # noqa: BLE001
                st.dataframe(m.style.format("{:.1%}"))
    with c2:
        try:
            import plotly.graph_objects as go

            eq = bt.equity.dropna()
            dd = eq / eq.cummax() - 1.0
            fig = go.Figure(
                go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", line=dict(color="#b22222"))
            )
            fig.update_layout(
                height=320,
                margin=dict(l=30, r=20, t=30, b=30),
                title="Drawdown",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:  # noqa: BLE001
            st.write("Drawdown chart unavailable.")

    # ---- factor research ---------------------------------------------------
    st.subheader("Factor Research")
    if state.factor_summary is not None and not state.factor_summary.empty:
        st.dataframe(state.factor_summary, use_container_width=True, height=320)
    else:
        st.info("No factor summary produced.")

    # ---- risk + attribution ------------------------------------------------
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.subheader("Risk Decomposition")
        if state.risk_result is not None:
            st.caption(
                f"Cross-sectional R² = {state.risk_result.r_squared:.3f} · "
                f"{state.risk_result.exposures.shape[1]} factors"
            )
            try:
                from alphaforge.risk.factor_model import factor_risk_decomposition

                w = bt.weights.abs().mean(axis=0)
                dec = factor_risk_decomposition(
                    w,
                    state.risk_result.exposures,
                    state.risk_result.factor_cov,
                    state.risk_result.specific_var,
                )
                st.dataframe(dec, use_container_width=True, height=300)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Risk decomposition table unavailable: {exc}")
        else:
            st.info("Risk model did not run.")

        st.subheader("Factor Attribution (returns-based)")
        if state.factor_attr is not None:
            fa = state.factor_attr
            fa_df = pd.DataFrame(
                {
                    "factor": fa.betas.index,
                    "beta": fa.betas.to_numpy(),
                    "attributed_return": fa.attributed_return.to_numpy(),
                }
            )
            st.dataframe(fa_df, use_container_width=True, height=220)
            st.caption(f"Attribution R² = {fa.r_squared:.3f}")
        else:
            st.info("Factor attribution did not run.")

    with rcol2:
        st.subheader("Brinson Attribution")
        if state.brinson is not None:
            b = state.brinson
            st.caption(
                f"Active {b.total_active:+.2%} · alloc {b.allocation:+.2%} · "
                f"sel {b.selection:+.2%} · inter {b.interaction:+.2%}"
            )
            try:
                import plotly.express as px

                bs = b.by_sector.copy()
                fig = px.bar(
                    bs, x="sector", y="active", title="Active return by sector", color="active"
                )
                fig.update_layout(
                    height=300, margin=dict(l=30, r=20, t=30, b=30), xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:  # noqa: BLE001
                st.dataframe(b.by_sector, use_container_width=True, height=300)
        else:
            st.info("Brinson attribution did not run.")

    # ---- copilot briefing --------------------------------------------------
    st.subheader("Research Copilot Briefing")
    if state.briefing is not None:
        st.success(state.briefing.headline)
        if state.briefing.findings:
            st.markdown("**Findings**")
            for f in state.briefing.findings:
                st.markdown(f"- {f}")
        if state.briefing.warnings:
            st.markdown("**Warnings**")
            for w in state.briefing.warnings:
                st.markdown(f"- ⚠️ {w}")
    else:
        st.info("Copilot briefing not available.")

    # ---- report link -------------------------------------------------------
    if state.report_path:
        st.divider()
        st.markdown(f"📄 Full self-contained HTML report: `{state.report_path}`")


if __name__ == "__main__":
    main()

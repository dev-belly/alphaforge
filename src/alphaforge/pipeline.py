"""End-to-end research pipeline.

This is the single object the CLI, the API and the notebooks call.  It runs the
whole stack in a fixed order, isolates each stage in try/except so a failure in
one layer still produces a report from the rest, and returns a ``state`` dict
that the tools, the report and the copilot all consume.

Order (every later stage only sees earlier, real outputs)::

    data -> panel -> factors -> dataset -> model -> risk -> portfolio -> backtest
         -> attribution (brinson + factor) -> report -> copilot briefing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.attribution import brinson_attribution, factor_attribution
from alphaforge.backtest.engine import BacktestConfig, BacktestEngine
from alphaforge.data.pipeline import DataPipeline
from alphaforge.factors import FactorContext
from alphaforge.features.fundamentals import FundamentalView
from alphaforge.features.panel import build_panel
from alphaforge.models.dataset import build_dataset
from alphaforge.models.pipeline import AlphaModelPipeline, signal_panel
from alphaforge.portfolio.constructor import PortfolioConstructor
from alphaforge.reporting import ReportInputs, write_report
from alphaforge.risk.factor_model import FundamentalRiskModel, RiskModelConfig
from alphaforge.risk.regime import (
    classify_regime,
    factor_performance_by_regime,
    regime_statistics,
)
from alphaforge.utils.config import Config
from alphaforge.utils.logging import Timer, get_logger

log = get_logger("pipeline")


@dataclass
class ResearchState:
    """Everything one run produces, in one place."""

    config: dict = field(default_factory=dict)
    panel: Any = None
    library: Any = None
    factor_summary: pd.DataFrame | None = None
    dataset: Any = None
    model_eval: Any = None
    signal_panel: pd.DataFrame | None = None
    risk_result: Any = None
    backtest: Any = None
    weights: pd.DataFrame | None = None
    brinson: Any = None
    factor_attr: Any = None
    regime: Any = None
    report_path: Path | None = None
    briefing: Any = None
    diagnostics: dict = field(default_factory=dict)

    def as_tool_state(self) -> dict:
        return {
            "library": self.library,
            "model_eval": self.model_eval,
            "backtest": self.backtest,
            "weights": self.weights,
            "risk_result": self.risk_result,
            "brinson": self.brinson,
            "factor_attr": self.factor_attr,
            "regime": self.regime,
            "config": self.config,
        }


class ResearchPipeline:
    """Runs the full stack and writes a report + briefing."""

    def __init__(self, config: Config | dict | None = None) -> None:
        self.config = (
            config
            if isinstance(config, Config)
            else Config.load()
            if config is None
            else Config(raw=dict(config))
        )

    # ------------------------------------------------------------------
    def run(
        self,
        start: str | None = None,
        end: str | None = None,
        model_type: str | None = None,
        report_dir: str = "research/reports",
        persist: bool = False,
    ) -> ResearchState:
        cfg = self.config.raw
        start = start or cfg.get("data", {}).get("start_date")
        end = end or cfg.get("data", {}).get("end_date")
        model_type = model_type or cfg.get("model", {}).get("type", "ridge")

        state = ResearchState(config=cfg)
        d = state.diagnostics

        # -- 1. data ----------------------------------------------------
        with Timer("pipeline.etl", log):
            res = DataPipeline(provider=cfg.get("data", {}).get("provider", "sample")).run(
                start=start,
                end=end,
                index_id=cfg.get("data", {}).get("universe", "SP500_SAMPLE"),
                persist=persist,
            )
        d["etl_symbols"] = int(res.bundle.prices["symbol"].nunique())

        # -- 2. panel ---------------------------------------------------
        benchmark = res.bundle.benchmark
        if benchmark is not None and not isinstance(benchmark, pd.Series):
            benchmark = pd.Series(benchmark)
        panel = build_panel(res.bundle.prices, universe=res.universe, benchmark=benchmark)
        state.panel = panel
        d["panel_dates"] = len(panel.dates)

        # -- 3. factors -------------------------------------------------
        fund = (
            FundamentalView.build(
                res.bundle.fundamentals, panel.dates, panel.symbols, panel.market_cap
            )
            if not res.bundle.fundamentals.empty
            else None
        )
        ctx = FactorContext(panel=panel, fundamentals=fund)
        from alphaforge.factors.library import FactorLibrary

        fl = FactorLibrary.from_config(ctx, cfg)
        fl.run()
        state.library = fl
        state.factor_summary = fl.summary_table()
        d["n_factors"] = len(fl.processed)

        # -- 4. dataset + model ---------------------------------------
        ds = build_dataset(
            panel,
            fl.processed,
            horizon=int(cfg.get("factor", {}).get("horizon", 21)),
            target=cfg.get("model", {}).get("target", "forward_rank"),
        )
        state.dataset = ds
        wf = AlphaModelPipeline.from_config(ds, cfg).run(model_type=model_type)
        state.model_eval = wf.evaluation
        state.signal_panel = signal_panel(wf.predictions, panel.dates, panel.symbols)
        d["rank_ic_mean"] = wf.evaluation.summary.get("rank_ic_mean")

        # -- 5. risk model ---------------------------------------------
        try:
            style_panels = _style_factor_exposures(
                fl.processed, cfg.get("risk", {}).get("style_factors", []), panel.symbols
            )
            rm = FundamentalRiskModel(RiskModelConfig.from_dict(cfg.get("risk", {})))
            state.risk_result = rm.fit(
                panel.returns, panel.market_cap.mean(axis=0), panel.industry.iloc[0], style_panels
            )
            d["risk_r2"] = state.risk_result.r_squared
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Risk model skipped: {exc}")

        # -- 6. backtest ------------------------------------------------
        cons = PortfolioConstructor(panel, cfg.get("portfolio", {}), cfg.get("risk", {}))
        bt = BacktestEngine(
            panel,
            constructor=cons,
            signals=state.signal_panel,
            ic=wf.evaluation.summary.get("rank_ic_mean", 0.03),
            config=BacktestConfig.from_dict(cfg.get("backtest", {})),
        ).run()
        state.backtest = bt
        state.weights = bt.weights
        d.update(bt.diagnostics)

        # -- 6b. market regime -----------------------------------------
        try:
            # Classify on the *full daily* benchmark (panel.benchmark) so there
            # is enough trailing history for the vol/trend windows. The backtest
            # benchmark is reindexed to monthly rebalance dates and is too short
            # (> 200 days are required by the classifier).
            bench_ret = (
                panel.benchmark.dropna()
                if getattr(panel, "benchmark", None) is not None
                else None
            )
            if bench_ret is not None and len(bench_ret) >= 200:
                state.regime = classify_regime(bench_ret)
                # Per-regime portfolio stats: align the daily regime labels onto
                # the (monthly) backtest return dates. Use exact-label match with
                # an as-of (last known regime) fallback for months whose exact
                # date is not in the daily benchmark index.
                bt_ret = bt.returns.dropna()
                reg_aligned = state.regime.reindex(bt_ret.index)
                miss = reg_aligned.isna()
                if bool(miss.any()):
                    reg_aligned = reg_aligned.fillna(state.regime.asof(bt_ret.index))
                d["regime_stats"] = regime_statistics(bt_ret, reg_aligned)
                if (
                    getattr(wf, "evaluation", None) is not None
                    and getattr(wf.evaluation, "ic_series", None) is not None
                ):
                    d["regime_factor_ic"] = factor_performance_by_regime(
                        wf.evaluation.ic_series, state.regime
                    )
                d["regime_label_counts"] = {
                    str(k): int(v) for k, v in state.regime.value_counts(dropna=True).items()
                }
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Market regime analysis skipped: {exc}")

        # -- 7. attribution --------------------------------------------
        try:
            bench_w = (
                res.universe.astype(float)
                .div(res.universe.sum(axis=1).replace(0, float("nan")), axis=0)
                .fillna(0.0)
            )
            state.brinson = brinson_attribution(
                bt.weights,
                bench_w.reindex(columns=panel.symbols).fillna(0.0),
                panel.returns.reindex(index=bt.weights.index, columns=panel.symbols).fillna(0.0),
                panel.industry,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Brinson attribution skipped: {exc}")

        try:
            fr = state.risk_result.factor_returns.reindex(bt.returns.index).dropna(how="all")
            if not fr.empty:
                state.factor_attr = factor_attribution(
                    bt.returns, fr, benchmark_returns=bt.benchmark
                )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Factor attribution skipped: {exc}")

        # -- 8. report --------------------------------------------------
        from alphaforge.agents.copilot import CopilotConfig, ResearchCopilot

        try:
            report_dir_p = Path(report_dir)
            report_dir_p.mkdir(parents=True, exist_ok=True)
            rpt = ReportInputs(
                title="AlphaForge Research Report",
                config=cfg,
                factor_summary=state.factor_summary,
                model_summary={
                    "quantile_returns": wf.evaluation.quantile_returns,
                    "ic_series": wf.evaluation.ic_series,
                },
                backtest=bt,
                risk_decomposition=(
                    state.risk_result.factor_cov if state.risk_result is not None else None
                ),
                brinson=state.brinson,
                factor_attribution=state.factor_attr,
                regime=state.regime,
                regime_stats=d.get("regime_stats"),
                notes=self._notes(state),
            )
            # attach covariance for the risk-contribution chart
            if state.risk_result is not None:
                rpt.risk_decomposition = _decomp_table(state)
            state.report_path = write_report(rpt, report_dir_p / "research_report.html")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Report generation skipped: {exc}")

        # -- 9. copilot ------------------------------------------------
        try:
            copilot = ResearchCopilot(CopilotConfig.from_dict(cfg))
            state.briefing = copilot.analyze(state.as_tool_state())
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Copilot briefing skipped: {exc}")

        log.info(f"Pipeline complete: {state.report_path}")
        return state

    # ------------------------------------------------------------------
    @staticmethod
    def _notes(state: ResearchState) -> list[str]:
        notes = []
        q = state.config.get("data", {})
        if q.get("survivorship_bias_disclaimer"):
            notes.append(
                "Survivorship bias disclaimer is ON: the sample provider supplies point-in-time membership only."
            )
        if state.backtest is not None:
            diag = state.backtest.diagnostics
            notes.append(
                f"Backtest: {diag.get('n_rebalances')} rebalances, "
                f"{diag.get('n_trades')} trades, avg turnover {diag.get('avg_turnover'):.2f}."
            )
        return notes


# Ordered candidate factor-library names per risk style factor. The first one
# present in the processed library wins; ``size`` is handled by market cap in
# the risk model itself, so it is intentionally omitted here.
_STYLE_ALIASES: dict[str, list[str]] = {
    "value": ["value_composite", "book_to_price", "earnings_yield", "ebit_to_ev"],
    "momentum": ["mom_12_1", "mom_120d", "residual_momentum", "mom_60d"],
    "volatility": ["volatility_60d", "volatility_252d", "downside_volatility"],
    "liquidity": ["amihud_illiquidity", "log_adv_21d", "dollar_volume_ratio"],
    "quality": ["quality_composite", "gross_profitability", "roa", "roe"],
}


def _style_factor_exposures(
    processed: dict[str, pd.DataFrame],
    style_factors: list[str],
    symbols: pd.Index,
) -> dict[str, pd.Series]:
    """Map risk style factors onto cross-sectional factor-library exposures.

    Each risk style factor expects a single cross-sectional Series indexed by
    symbol. We take the time-mean of the best-matching processed factor panel so
    the exposures are stable snapshots the risk model standardises internally.
    """
    out: dict[str, pd.Series] = {}
    for sf in style_factors:
        if sf == "size":
            continue  # handled by market cap inside the risk model
        for cand in _STYLE_ALIASES.get(sf, []):
            if cand in processed:
                panel_for_style = processed[cand]
                if isinstance(panel_for_style, pd.DataFrame):
                    series = panel_for_style.mean(axis=0)
                else:
                    series = panel_for_style
                out[sf] = series.reindex(symbols)
                break
    return out


def _decomp_table(state: ResearchState):
    """Risk decomposition table with the covariance attached for the chart."""
    rm = state.risk_result
    if rm is None:
        return None
    from alphaforge.risk.factor_model import factor_risk_decomposition

    try:
        w = state.weights.abs().mean(axis=0) if state.weights is not None else None
        if w is None:
            return rm.factor_cov
        table = factor_risk_decomposition(w, rm.exposures, rm.factor_cov, rm.specific_var)
        table.attrs["covariance"] = rm.covariance
        return table
    except Exception:  # noqa: BLE001
        return rm.factor_cov


def run_research(config: dict | None = None, **kwargs) -> ResearchState:
    """Module-level convenience entry used by the CLI and scripts."""
    return ResearchPipeline(config).run(**kwargs)


__all__ = ["ResearchPipeline", "ResearchState", "run_research"]

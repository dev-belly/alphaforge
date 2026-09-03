"""Deterministic tool layer for the research copilot.

These functions are the *only* things the copilot is allowed to read.  They
wrap the real outputs of every upstream stage and return plain Python objects,
never prose.  The copilot reasons over the returned numbers; it never invents
them.  If a stage was not run, the tool returns ``None`` and the copilot says so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("agents.tools")


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: object
    note: str = ""


# --------------------------------------------------------------------------
def factor_summary_table(library) -> ToolResult:
    """Top factors by information ratio / selection reach."""
    try:
        df = library.summary_table()
        df = df.sort_values("icir", ascending=False).head(15)
        return ToolResult("factors", True, df, f"{len(df)} factors ranked by ICIR")
    except Exception as exc:  # noqa: BLE001
        return ToolResult("factors", False, None, str(exc))


def model_evaluation(model_eval) -> ToolResult:
    """Out-of-sample model quality: IC, ICIR, turnover, long-short spread."""
    if model_eval is None:
        return ToolResult("model", False, None, "model not run")
    s = dict(model_eval.summary)
    return ToolResult(
        "model",
        True,
        s,
        f"rank_ic={s.get('rank_ic_mean'):+.4f} icir={s.get('icir'):+.3f} turnover={s.get('turnover'):.2f}",
    )


def backtest_metrics(backtest) -> ToolResult:
    """Headline backtest statistics."""
    if backtest is None:
        return ToolResult("backtest", False, None, "backtest not run")
    return ToolResult("backtest", True, dict(backtest.summary()), "summary metrics")


def backtest_diagnostics(backtest) -> ToolResult:
    if backtest is None:
        return ToolResult("diagnostics", False, None, "backtest not run")
    return ToolResult("diagnostics", True, dict(backtest.diagnostics), "run diagnostics")


def risk_decomposition(weights: pd.Series | pd.DataFrame, risk_result) -> ToolResult:
    """Factor and specific risk split, plus per-factor exposure."""
    if risk_result is None:
        return ToolResult("risk", False, None, "risk model not run")
    try:
        # The pipeline stores weights as a (assets x dates) DataFrame; the agent
        # only needs the latest rebalance's book. Normalise either input shape.
        if isinstance(weights, pd.DataFrame):
            weights = weights.iloc[:, -1]
        rc = weights.reindex(risk_result.exposures.index).fillna(0.0)
        decomp = {
            "r_squared": float(risk_result.r_squared),
            "n_factors": int(risk_result.covariance.shape[0]),
            "top_exposures": rc.abs().sort_values(ascending=False).head(8).to_dict(),
        }
        return ToolResult("risk", True, decomp, f"risk-model R^2={risk_result.r_squared:.3f}")
    except Exception as exc:  # noqa: BLE001
        return ToolResult("risk", False, None, str(exc))


def attribution_summary(brinson, factor_attr) -> ToolResult:
    """Where active return came from and what risk drove it."""
    out: dict = {}
    if brinson is not None:
        out["brinson"] = brinson.to_dict()
    if factor_attr is not None:
        out["factor"] = factor_attr.to_dict()
    if not out:
        return ToolResult("attribution", False, None, "no attribution run")
    return ToolResult("attribution", True, out, "sector + factor attribution")


def data_quality(quality) -> ToolResult:
    """ETL quality gates and survivorship flags."""
    if quality is None:
        return ToolResult("quality", False, None, "no data report")
    try:
        d = quality.to_dict()
        return ToolResult("quality", True, d, f"coverage={d.get('coverage')}")
    except Exception as exc:  # noqa: BLE001
        return ToolResult("quality", False, None, str(exc))


def config_snapshot(config: dict) -> ToolResult:
    return ToolResult("config", True, config, "effective configuration")


def stress_test(stress: dict) -> ToolResult:
    """Scenario P&L book: per-scenario portfolio loss under factor shocks."""
    if not stress:
        return ToolResult("stress", False, None, "stress not run")
    try:
        data = {nm: r.to_dict() for nm, r in stress.items()}
        worst = min(stress.values(), key=lambda r: r.pnl_pct)
        return ToolResult(
            "stress",
            True,
            data,
            f"{len(stress)} scenarios; worst {worst.scenario} {worst.pnl_pct:+.2%}",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult("stress", False, None, str(exc))


def analyze_market_regime(regime: pd.Series, backtest: Any = None) -> ToolResult:
    """Cross-sectional regime split: label counts + per-regime portfolio stats.

    ``regime`` is the per-date label Series produced by the pipeline.  When the
    backtest object is supplied its realised return series is split by regime so
    the copilot can report how the strategy behaved in Bull/Bear x High/Low-Vol
    regimes.
    """
    if regime is None:
        return ToolResult("regime", False, None, "regime not run")
    try:
        from alphaforge.risk.regime import REGIME_LABELS, regime_statistics

        counts = {str(k): int(v) for k, v in regime.value_counts(dropna=True).items()}
        data: dict = {"labels": list(REGIME_LABELS), "counts": counts}
        if backtest is not None and getattr(backtest, "returns", None) is not None:
            rets = backtest.returns
            rets = rets.dropna() if isinstance(rets, pd.Series) else rets
            # Daily regime labels -> (monthly) backtest returns: align by exact
            # date, fall back to last known regime as-of that date.
            reg_aligned = regime.reindex(rets.index)
            miss = reg_aligned.isna()
            if bool(miss.any()):
                reg_aligned = reg_aligned.fillna(regime.asof(rets.index))
            data["return_stats"] = regime_statistics(rets, reg_aligned)
        return ToolResult(
            "regime",
            True,
            data,
            f"regimes: {counts}",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult("regime", False, None, str(exc))


# --------------------------------------------------------------------------
CATALOG: dict[str, Callable] = {
    "factors": factor_summary_table,
    "model": model_evaluation,
    "backtest": backtest_metrics,
    "diagnostics": backtest_diagnostics,
    "risk": risk_decomposition,
    "attribution": attribution_summary,
    "regime": analyze_market_regime,
    "stress": stress_test,
    "quality": data_quality,
    "config": config_snapshot,
}


def run_tools(state: dict) -> dict[str, ToolResult]:
    """Run every tool whose inputs are present in ``state``.

    ``state`` keys mirror the pipeline outputs: ``library``, ``model_eval``,
    ``backtest``, ``weights``, ``risk_result``, ``brinson``, ``factor_attr``,
    ``regime``, ``quality``, ``config``.
    """
    results: dict[str, ToolResult] = {}
    if "library" in state:
        results["factors"] = factor_summary_table(state["library"])
    if "model_eval" in state:
        results["model"] = model_evaluation(state["model_eval"])
    if "backtest" in state:
        results["backtest"] = backtest_metrics(state["backtest"])
        results["diagnostics"] = backtest_diagnostics(state["backtest"])
    if "risk_result" in state and "weights" in state:
        results["risk"] = risk_decomposition(state["weights"], state["risk_result"])
    if "brinson" in state or "factor_attr" in state:
        results["attribution"] = attribution_summary(state.get("brinson"), state.get("factor_attr"))
    if "regime" in state:
        results["regime"] = analyze_market_regime(state["regime"], state.get("backtest"))
    if "stress" in state:
        results["stress"] = stress_test(state["stress"])
    if "quality" in state:
        results["quality"] = data_quality(state["quality"])
    if "config" in state:
        results["config"] = config_snapshot(state["config"])
    return results


__all__ = ["ToolResult", "run_tools", "CATALOG"]

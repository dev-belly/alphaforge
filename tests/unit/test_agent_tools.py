"""Fast unit tests for the deterministic copilot tool layer.

These pin the agent tools without spinning up the (slow) full pipeline, so a
regression in e.g. ``risk_decomposition`` fails the fast gate instead of only
the slow integration job.  Every tool in ``CATALOG`` is exercised here: the
``None``/missing-input guard branch *and* the happy-path extraction, plus an
end-to-end ``run_tools()`` that drives the whole catalog with a correctly
shaped ``state``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.agents.tools import (
    CATALOG,
    analyze_market_regime,
    attribution_summary,
    backtest_diagnostics,
    backtest_metrics,
    config_snapshot,
    data_quality,
    factor_summary_table,
    model_evaluation,
    risk_decomposition,
    run_tools,
    stress_test,
)
from alphaforge.risk.factor_model import RiskModelResult
from alphaforge.risk.regime import REGIME_LABELS


# --- lightweight fakes that match the shapes the real pipeline emits -------
class _FakeObj:
    """Stand-in carrying an arbitrary dict, exposing both ``.to_dict()``."""

    def __init__(self, **kw: object) -> None:
        self._d = kw

    def to_dict(self) -> dict:
        return dict(self._d)


class _FakeModelEval:
    summary = {"rank_ic_mean": 0.0447, "icir": 0.211, "turnover": 0.50}


class _FakeBacktest:
    def summary(self) -> dict:
        return {"cagr": 0.0079, "sharpe": 0.13, "max_drawdown": -0.228}

    diagnostics = {"n_periods": 2268, "lookahead_flags": 0}
    returns = None


class _FakeLibrary:
    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "factor": ["mom_60d", "value", "size"],
                "icir": [0.30, 0.18, 0.05],
                "rank_ic_mean": [0.05, 0.03, 0.01],
            }
        )


class _FakeStress:
    def __init__(self, scenario: str, pnl_pct: float) -> None:
        self.scenario = scenario
        self.pnl_pct = pnl_pct

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "pnl_pct": self.pnl_pct}


def _make_risk_result() -> RiskModelResult:
    assets = ["AAA", "BBB", "CCC", "DDD"]
    factors = ["size", "value", "momentum"]
    exposures = pd.DataFrame(
        np.array(
            [
                [0.8, -0.3, 0.5],
                [0.7, 0.2, -0.4],
                [-0.6, 0.5, 0.3],
                [0.1, -0.2, 0.6],
            ]
        ),
        index=assets,
        columns=factors,
    )
    cov = pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
    factor_cov = pd.DataFrame(np.eye(len(factors)), index=factors, columns=factors)
    return RiskModelResult(
        exposures=exposures,
        factor_cov=factor_cov,
        specific_var=pd.Series(0.01, index=assets),
        covariance=cov,
        factor_returns=pd.DataFrame(
            0.0, index=pd.date_range("2024-01-01", periods=3), columns=factors
        ),
        residuals=pd.DataFrame(0.0, index=assets, columns=pd.date_range("2024-01-01", periods=3)),
        r_squared=0.51,
    )


# --------------------------------------------------------------------------
# risk (the originally-buggy tool)
def test_risk_decomposition_accepts_series_weights() -> None:
    rr = _make_risk_result()
    w = pd.Series([0.3, 0.3, 0.2, 0.2], index=rr.exposures.index)
    res = risk_decomposition(w, rr)
    assert res.ok
    assert "top_exposures" in res.data
    assert res.data["r_squared"] == 0.51


def test_risk_decomposition_accepts_dataframe_weights() -> None:
    # Regression: the pipeline stores weights as a (assets x dates) DataFrame,
    # not a Series. The tool must normalise it to the latest rebalance column,
    # otherwise .abs().sort_values(ascending=False) raises ValueError on a
    # DataFrame (needs 'by') and the copilot's /agent/query tool=risk 404s.
    rr = _make_risk_result()
    w = pd.DataFrame(
        {
            pd.Timestamp("2024-01-01"): [0.3, 0.3, 0.2, 0.2],
            pd.Timestamp("2024-02-01"): [0.25, 0.25, 0.25, 0.25],
        },
        index=rr.exposures.index,
    )
    res = risk_decomposition(w, rr)
    assert res.ok, res.note
    assert "top_exposures" in res.data


def test_risk_decomposition_no_model_is_false() -> None:
    w = pd.Series([0.25, 0.25, 0.25, 0.25], index=["AAA", "BBB", "CCC", "DDD"])
    res = risk_decomposition(w, None)
    assert not res.ok


# --------------------------------------------------------------------------
# factors
def test_factor_summary_table_ranks_by_icir() -> None:
    res = factor_summary_table(_FakeLibrary())
    assert res.ok
    icir = list(res.data["icir"])
    assert icir == sorted(icir, reverse=True)
    assert len(res.data) <= 15


def test_factor_summary_table_bad_input_is_false() -> None:
    # No explicit None-guard: a bad input must surface as ok=False, not raise.
    res = factor_summary_table(None)
    assert not res.ok


# --------------------------------------------------------------------------
# model
def test_model_evaluation_none_is_false() -> None:
    res = model_evaluation(None)
    assert not res.ok
    assert "not run" in res.note


def test_model_evaluation_reports_metrics() -> None:
    res = model_evaluation(_FakeModelEval())
    assert res.ok
    assert res.data["rank_ic_mean"] == 0.0447
    assert "rank_ic" in res.note


# --------------------------------------------------------------------------
# backtest + diagnostics
def test_backtest_metrics_none_is_false() -> None:
    assert not backtest_metrics(None).ok


def test_backtest_metrics_summary() -> None:
    res = backtest_metrics(_FakeBacktest())
    assert res.ok
    assert res.data["sharpe"] == 0.13


def test_backtest_diagnostics_none_is_false() -> None:
    assert not backtest_diagnostics(None).ok


def test_backtest_diagnostics_returns_dict() -> None:
    res = backtest_diagnostics(_FakeBacktest())
    assert res.ok
    assert res.data["n_periods"] == 2268


# --------------------------------------------------------------------------
# attribution
def test_attribution_summary_empty_is_false() -> None:
    res = attribution_summary(None, None)
    assert not res.ok
    assert "no attribution" in res.note


def test_attribution_summary_merges_brinson_and_factor() -> None:
    res = attribution_summary(_FakeObj(sector=0.01), _FakeObj(factor=0.02))
    assert res.ok
    assert "brinson" in res.data and "factor" in res.data


# --------------------------------------------------------------------------
# quality
def test_data_quality_none_is_false() -> None:
    assert not data_quality(None).ok


def test_data_quality_report() -> None:
    res = data_quality(_FakeObj(coverage=0.98, missing=12))
    assert res.ok
    assert res.data["coverage"] == 0.98


# --------------------------------------------------------------------------
# config
def test_config_snapshot_echoes_config() -> None:
    cfg = {"model": "ridge", "start": "2016-01-01"}
    res = config_snapshot(cfg)
    assert res.ok
    assert res.data == cfg


# --------------------------------------------------------------------------
# stress
def test_stress_test_empty_is_false() -> None:
    assert not stress_test({}).ok


def test_stress_test_reports_worst_scenario() -> None:
    stress = {
        "rate_shock": _FakeStress("rate_shock", -0.04),
        "equity_crash": _FakeStress("equity_crash", -0.15),
    }
    res = stress_test(stress)
    assert res.ok
    assert res.data["equity_crash"]["pnl_pct"] == -0.15
    assert "worst equity_crash" in res.note


# --------------------------------------------------------------------------
# regime
def test_analyze_market_regime_none_is_false() -> None:
    assert not analyze_market_regime(None).ok


def test_analyze_market_regime_counts_labels() -> None:
    labels = ["Bull/LowVol", "Bear/HighVol", "Bull/LowVol", "Bear/HighVol"]
    res = analyze_market_regime(pd.Series(labels))
    assert res.ok
    assert res.data["counts"]["Bull/LowVol"] == 2
    assert set(res.data["labels"]) == set(REGIME_LABELS)


# --------------------------------------------------------------------------
# end-to-end: every catalog tool runs green from one state
def test_run_tools_covers_full_catalog() -> None:
    rr = _make_risk_result()
    w = pd.Series([0.25, 0.25, 0.25, 0.25], index=rr.exposures.index)
    state = {
        "library": _FakeLibrary(),
        "model_eval": _FakeModelEval(),
        "backtest": _FakeBacktest(),
        "weights": w,
        "risk_result": rr,
        "brinson": _FakeObj(sector=0.01),
        "factor_attr": _FakeObj(factor=0.02),
        "regime": pd.Series(["Bull/LowVol", "Bear/HighVol"]),
        "stress": {"crash": _FakeStress("crash", -0.10)},
        "quality": _FakeObj(coverage=0.99),
        "config": {"model": "ridge"},
    }
    results = run_tools(state)
    assert set(results.keys()) == set(CATALOG.keys())
    for name, r in results.items():
        assert r.ok, f"{name} -> {r.note}"

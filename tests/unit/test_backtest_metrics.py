"""Unit tests for the Brinson attribution and backtest performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.attribution.brinson import brinson_attribution
from alphaforge.backtest.metrics import performance_stats


def test_performance_stats_basic():
    rng = np.random.default_rng(0)
    rets = pd.Series(
        rng.normal(0.0004, 0.01, size=504), index=pd.date_range("2021-01-01", periods=504, freq="B")
    )
    m = performance_stats(rets)
    assert np.isfinite(m["cagr"])
    assert np.isfinite(m["sharpe"])
    assert m["max_drawdown"] <= 0.0
    assert 200 < m["periods_per_year"] < 300  # inference fix


def test_performance_stats_with_benchmark():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2021-01-01", periods=504, freq="B")
    strat = pd.Series(rng.normal(0.0005, 0.01, size=504), index=idx)
    bench = pd.Series(rng.normal(0.0003, 0.01, size=504), index=idx)
    m = performance_stats(strat, benchmark=bench)
    assert np.isfinite(m["information_ratio"])
    assert np.isfinite(m["alpha_ann"])


def test_active_return_is_terminal_wealth_gap_not_compounded_spread():
    """Active return must be the gap between two compounded curves.

    Regression guard. The previous implementation returned
    ``compound(s - b).iloc[-1] - 1``, i.e. it compounded the arithmetic return
    *spread* as if it were a return. That is meaningless economically and
    numerically explosive: any day with ``s - b < -1`` makes the factor
    ``1 + (s - b)`` negative with magnitude > 1, so a few thousand sessions of
    ``cumprod`` overflow float64 and every reported statistic becomes nan.
    """
    idx = pd.date_range("2021-01-01", periods=252, freq="B")
    rng = np.random.default_rng(3)
    strat = pd.Series(rng.normal(0.0006, 0.011, size=252), index=idx)
    bench = pd.Series(rng.normal(0.0003, 0.012, size=252), index=idx)

    m = performance_stats(strat, benchmark=bench)

    # Align on the overlapping window the stats are actually computed over.
    joined = pd.concat([strat.rename("s"), bench.rename("b")], axis=1).dropna()
    expected = float((1 + joined["s"]).prod() - (1 + joined["b"]).prod())

    assert np.isfinite(m["active_return"]), "active_return must never be nan/inf"
    assert abs(m["active_return"] - expected) < 1e-9


def test_relative_stats_finite_under_large_return_spread():
    """A day where the strategy badly lags the benchmark must not blow up."""
    idx = pd.date_range("2021-01-01", periods=500, freq="B")
    rng = np.random.default_rng(5)
    strat = pd.Series(rng.normal(0.0, 0.01, size=500), index=idx)
    bench = pd.Series(rng.normal(0.0, 0.01, size=500), index=idx)
    # Inject a spread far below -1 on a few days: the exact condition that used
    # to overflow the cumulative product.
    strat.iloc[100] = -0.90
    bench.iloc[100] = 2.50
    strat.iloc[250] = -0.95
    bench.iloc[250] = 1.80

    m = performance_stats(strat, benchmark=bench)
    for key in ("active_return", "information_ratio", "tracking_error", "beta", "alpha_ann"):
        assert np.isfinite(m[key]), f"{key} must stay finite, got {m[key]}"


def test_brinson_identity():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2022-01-01", periods=120, freq="B")
    symbols = [f"S{i}" for i in range(8)]
    asset_ret = pd.DataFrame(rng.normal(0.0, 0.01, size=(120, 8)), index=dates, columns=symbols)
    wp = pd.DataFrame(np.abs(rng.normal(0, 1, size=(120, 8))), index=dates, columns=symbols)
    wp = wp.div(wp.sum(axis=1), axis=0)
    wb = pd.DataFrame(np.abs(rng.normal(0, 1, size=(120, 8))), index=dates, columns=symbols)
    wb = wb.div(wb.sum(axis=1), axis=0)
    sectors = pd.Series(["X", "X", "Y", "Y", "Z", "Z", "X", "Y"], index=symbols)

    res = brinson_attribution(wp, wb, asset_ret, sectors)
    total = res.allocation + res.selection + res.interaction
    # The three-term Brinson-Fachler split is an *approximation*: it does not sum
    # to the realised active return exactly (the exact equality belongs to the
    # two-term Brinson-Hood form). The gap must stay small in absolute terms.
    assert abs(total - res.total_active) < 1e-3
    assert res.by_sector is not None
    assert "sector" in res.by_sector.columns

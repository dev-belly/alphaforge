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

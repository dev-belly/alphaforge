"""Regression tests for benchmark handling in the backtest engine.

These guard the look-ahead / double-differencing class of bug where a benchmark
that already travels as a *return* series was differenced a second time
(``bench.pct_change()`` on an already-return series). That turned a ~15%-vol
benchmark into a 58,000%-vol one and silently destroyed beta, alpha, tracking
error, information ratio and every capture ratio derived from it.

See ``alphaforge.utils.math_utils.ensure_returns`` and the engine boundary in
``alphaforge.backtest.engine.BacktestEngine.run``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.backtest.engine import BacktestConfig, BacktestEngine
from alphaforge.features.panel import MarketPanel
from alphaforge.utils.math_utils import TRADING_DAYS_PER_YEAR, ensure_returns


def _ret_vol(series: pd.Series) -> float:
    return float(series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _make_panel(
    dates: pd.DatetimeIndex, assets: list[str], rng: np.random.Generator, benchmark: pd.Series
) -> MarketPanel:
    """A minimal but internally-consistent MarketPanel for engine smoke checks."""
    n = len(assets)
    market = rng.normal(0.0003, 0.009, size=len(dates))
    beta = rng.uniform(0.5, 1.2, size=n)
    noise = rng.normal(0.0, 0.011, size=(len(dates), n))
    arr = market[:, None] * beta[None, :] + noise
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + arr, axis=0), index=dates, columns=assets)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, size=(len(dates), n)), index=dates, columns=assets)
    market_cap = pd.DataFrame(
        rng.uniform(1e9, 1e11, size=(len(dates), n)), index=dates, columns=assets
    )
    universe = pd.DataFrame(True, index=dates, columns=assets)
    industry = pd.DataFrame(
        np.tile(np.resize(["Tech", "Fin", "Energy", "Health"], n), (len(dates), 1)),
        index=dates,
        columns=assets,
    )
    returns = close.pct_change(fill_method=None)
    dollar_volume = close * volume
    return MarketPanel(
        dates=pd.DatetimeIndex(dates),
        close=close,
        raw_close=close,
        returns=returns,
        volume=volume,
        dollar_volume=dollar_volume,
        market_cap=market_cap,
        universe=universe,
        industry=industry,
        benchmark=benchmark,
    )


def test_ensure_returns_preserves_return_series():
    """A benchmark that is already a return series must pass through unchanged."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-01-01", periods=504, freq="B")
    true_vol = 0.15
    bench = pd.Series(
        rng.normal(0.0004, true_vol / np.sqrt(252), size=504), index=idx, name="benchmark"
    )
    out = ensure_returns(bench)
    assert len(out) >= 500
    # No re-differentiation: volatility and typical magnitude are preserved.
    assert abs(_ret_vol(out) - true_vol) < 0.02
    assert out.abs().median() < 0.1  # still a return series, not a level


def test_ensure_returns_no_double_difference_on_returns():
    """The exact historical bug, encoded: pct_change() explodes; ensure_returns does not."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2021-01-01", periods=504, freq="B")
    bench = pd.Series(
        rng.normal(0.0004, 0.15 / np.sqrt(252), size=504), index=idx, name="benchmark"
    )
    buggy = bench.pct_change(fill_method=None).dropna()
    fixed = ensure_returns(bench)
    buggy_vol = _ret_vol(buggy)
    fixed_vol = _ret_vol(fixed)
    assert buggy_vol > 1.0, "old code should explode past 100% annualised vol"
    assert fixed_vol < 0.25, "fix should keep benchmark vol sane"


def test_ensure_returns_converts_price_level():
    """A raw price/index/NAV level handed in directly must be converted once."""
    rng = np.random.default_rng(13)
    idx = pd.date_range("2021-01-01", periods=504, freq="B")
    true_vol = 0.18
    rets = rng.normal(0.0004, true_vol / np.sqrt(252), size=504)
    level = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx, name="level")
    out = ensure_returns(level)
    assert abs(_ret_vol(out) - true_vol) < 0.03


def test_backtest_benchmark_vol_sane_with_return_series():
    """End-to-end: running a backtest with a return-series benchmark must yield
    sane beta / alpha / tracking error, not the silently-corrupted values the
    double-differencing bug produced."""
    rng = np.random.default_rng(21)
    dates = pd.date_range("2020-01-01", periods=320, freq="B")
    assets = [f"A{i:02d}" for i in range(20)]
    true_vol = 0.15
    bench_rets = pd.Series(
        rng.normal(0.0003, true_vol / np.sqrt(252), size=320), index=dates, name="benchmark"
    )
    panel = _make_panel(dates, assets, rng, bench_rets)

    def equal_weight(_date, _prev):
        return pd.Series(1.0 / len(assets), index=assets)

    cfg = BacktestConfig(rebalance="monthly", min_history_days=252, initial_capital=1e7)
    res = BacktestEngine(panel=panel, weight_fn=equal_weight, config=cfg).run()

    assert res.benchmark is not None
    assert np.isfinite(res.metrics["benchmark_vol"])
    # The headline guard: benchmark vol must be in the right ballpark, never
    # the 58,000%-vol artefact.
    assert res.metrics["benchmark_vol"] < 0.25
    assert abs(res.metrics["benchmark_vol"] - true_vol) < 0.04
    # Relative stats derived from a sane benchmark must be finite and reasonable.
    assert np.isfinite(res.metrics["beta"])
    assert abs(res.metrics["beta"]) < 3.0
    assert np.isfinite(res.metrics["alpha_ann"])
    assert np.isfinite(res.metrics["tracking_error"])
    assert np.isfinite(res.metrics["information_ratio"])

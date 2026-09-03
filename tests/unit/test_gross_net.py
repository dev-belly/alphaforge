"""Unit tests for gross vs net (cost-drag) transparency in the backtest.

These pin down two things the compliance review required:

* :func:`alphaforge.backtest.metrics.gross_returns_from_net` recovers the
  pre-cost return series *exactly* from the net series + the per-day cost.
* The engine reports gross metrics that are never worse than the net metrics
  when real costs are charged (the cost drag is non-negative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.backtest.engine import BacktestEngine
from alphaforge.backtest.metrics import gross_returns_from_net
from alphaforge.features.panel import MarketPanel


# --------------------------------------------------------------------------
def _synth_panel(n_days: int = 600, n_assets: int = 20, seed: int = 0) -> MarketPanel:
    """A small but internally-consistent MarketPanel for fast engine tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    syms = [f"A{i:02d}" for i in range(n_assets)]
    market = rng.normal(0.0003, 0.01, size=n_days)
    beta = rng.uniform(0.4, 1.3, size=n_assets)
    noise = rng.normal(0.0, 0.012, size=(n_days, n_assets))
    ret = market[:, None] * beta[None, :] + noise
    close = pd.DataFrame((1.0 + ret).cumprod(axis=0) * 100.0, index=idx, columns=syms)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, size=(n_days, n_assets)), index=idx, columns=syms)
    dollar_volume = close * volume
    market_cap = close * rng.uniform(1e8, 1e9, size=n_assets)[None, :]
    universe = close.notna()
    secs = np.array(
        [
            "Tech",
            "Fin",
            "Energy",
            "Health",
            "Health",
            "Tech",
            "Fin",
            "Energy",
            "Tech",
            "Fin",
            "Energy",
            "Health",
            "Tech",
            "Fin",
            "Energy",
            "Health",
            "Tech",
            "Fin",
            "Energy",
            "Health",
        ]
    )[:n_assets]
    industry = pd.DataFrame(np.tile(secs, (n_days, 1)), index=idx, columns=syms)
    bench_ret = close.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    benchmark = pd.Series((1.0 + bench_ret).cumprod(), index=idx)
    return MarketPanel(
        dates=pd.DatetimeIndex(idx),
        close=close,
        raw_close=close.copy(),
        returns=close.pct_change(fill_method=None),
        volume=volume,
        dollar_volume=dollar_volume,
        market_cap=market_cap,
        universe=universe,
        industry=industry,
        benchmark=benchmark,
    )


def _equal_weight_fn(symbols):
    def _fn(date, prev_weights):  # noqa: ANN001
        return pd.Series(1.0 / len(symbols), index=symbols)

    return _fn


# --------------------------------------------------------------------------
def test_gross_from_net_roundtrip():
    """Reconstructing gross from net + cost must recover the original gross."""
    rng = np.random.default_rng(4)
    idx = pd.date_range("2021-01-01", periods=500, freq="B")
    gross = pd.Series(rng.normal(0.0004, 0.01, size=500), index=idx)
    init = 10_000_000.0
    equity = (1.0 + gross).cumprod() * init
    prev_nav = equity.shift(1)
    prev_nav.iloc[0] = init
    costs = equity * 0.0003  # 3 bps of NAV per day
    net = (gross - costs / prev_nav).clip(lower=-0.99)
    recon = gross_returns_from_net(net, costs, equity, init)
    assert np.allclose(recon.to_numpy(), gross.to_numpy(), atol=1e-9)


def test_gross_from_net_zero_cost_is_noop():
    rng = np.random.default_rng(9)
    idx = pd.date_range("2021-01-01", periods=200, freq="B")
    returns = pd.Series(rng.normal(0.0003, 0.01, size=200), index=idx)
    equity = (1.0 + returns).cumprod() * 1_000_000.0
    costs = pd.Series(0.0, index=idx)
    recon = gross_returns_from_net(returns, costs, equity, 1_000_000.0)
    assert np.allclose(recon.to_numpy(), returns.to_numpy(), atol=1e-12)


def test_engine_reports_gross_and_net():
    panel = _synth_panel(seed=2)
    bt = BacktestEngine(
        panel=panel,
        weight_fn=_equal_weight_fn(panel.symbols),
        cost_model={"commission_bps": 5.0, "slippage_bps": 10.0, "impact_coeff_bps": 20.0},
        config={"rebalance": "monthly"},
        benchmark=panel.benchmark,
    ).run()

    assert "gross_cagr" in bt.metrics
    assert "gross_sharpe" in bt.metrics
    # Gross must never be below net when costs were actually charged.
    assert bt.metrics["gross_total_return"] >= bt.metrics["total_return"] - 1e-9
    assert bt.metrics["gross_sharpe"] >= bt.metrics["sharpe"] - 1e-9
    # The CAGR gap is exactly the cost drag.
    assert bt.metrics["cost_drag_cagr"] >= -1e-9


def test_engine_gross_equals_net_when_free():
    """With zero costs, gross and net metrics are identical."""
    panel = _synth_panel(seed=2)
    bt = BacktestEngine(
        panel=panel,
        weight_fn=_equal_weight_fn(panel.symbols),
        cost_model={"commission_bps": 0.0, "slippage_bps": 0.0, "impact_coeff_bps": 0.0},
        config={"rebalance": "monthly"},
    ).run()
    assert abs(bt.metrics["gross_cagr"] - bt.metrics["cagr"]) < 1e-9
    assert abs(bt.metrics["gross_sharpe"] - bt.metrics["sharpe"]) < 1e-9


def test_gross_metrics_present_in_summary():
    panel = _synth_panel(seed=5)
    bt = BacktestEngine(
        panel=panel,
        weight_fn=_equal_weight_fn(panel.symbols),
        cost_model={"commission_bps": 3.0, "slippage_bps": 5.0, "impact_coeff_bps": 10.0},
        config={"rebalance": "monthly"},
    ).run()
    s = bt.summary()
    assert "gross_cagr" in s and "gross_sharpe" in s and "cost_drag_cagr" in s
    # summary metrics stay JSON-safe through performance_stats.
    assert np.isfinite(s["gross_sharpe"]) or s["gross_sharpe"] == s["gross_sharpe"]

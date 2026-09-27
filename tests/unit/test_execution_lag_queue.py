"""Signals must retain their own execution dates when schedules overlap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtest.engine import BacktestConfig, BacktestEngine
from alphaforge.features.panel import MarketPanel


def test_daily_signals_execute_exactly_two_sessions_later() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    symbols = ["A", "B"]
    close = pd.DataFrame(100.0, index=dates, columns=symbols)
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=symbols)
    panel = MarketPanel(
        dates=dates,
        close=close,
        raw_close=close.copy(),
        returns=close.pct_change(fill_method=None),
        volume=volume,
        dollar_volume=close * volume,
        market_cap=pd.DataFrame(1e9, index=dates, columns=symbols),
        universe=pd.DataFrame(True, index=dates, columns=symbols),
        industry=pd.DataFrame("Other", index=dates, columns=symbols),
    )

    def alternate(signal_date: pd.Timestamp, _previous: pd.Series | None) -> pd.Series:
        i = dates.get_loc(signal_date)
        return pd.Series({"A": float(i % 2 == 0), "B": float(i % 2 == 1)})

    result = BacktestEngine(
        panel,
        weight_fn=alternate,
        config=BacktestConfig(rebalance="daily", min_history_days=2, execution_lag_days=2),
        cost_model={"commission_bps": 0.0, "slippage_bps": 0.0, "impact_coeff_bps": 0.0},
    ).run()

    # The first tradable signal is generated at index 2. At index 4 it must
    # still be that signal, even though another one arrived at index 3.
    for execution_index in range(4, len(dates)):
        expected = "A" if (execution_index - 2) % 2 == 0 else "B"
        actual = result.weights.loc[dates[execution_index], expected]
        assert np.isclose(actual, 1.0), (execution_index, expected, actual)
        trades = result.trades.loc[result.trades["date"] == dates[execution_index]]
        assert not trades.empty
        assert trades["signal_date"].eq(dates[execution_index - 2]).all()
    assert result.diagnostics["n_rebalances"] == len(dates) - 4
    assert result.diagnostics["n_unexecuted_signals"] == 2


def test_same_close_execution_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one trading session"):
        BacktestConfig(execution_lag_days=0)

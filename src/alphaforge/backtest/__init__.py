"""Backtesting: performance analytics and the event-driven engine."""

from alphaforge.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult, run_backtest
from alphaforge.backtest.metrics import (
    MetricsConfig,
    drawdown_table,
    monthly_returns,
    performance_stats,
    rolling_metrics,
    summarise,
    yearly_returns,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "MetricsConfig",
    "performance_stats",
    "monthly_returns",
    "yearly_returns",
    "rolling_metrics",
    "drawdown_table",
    "summarise",
]

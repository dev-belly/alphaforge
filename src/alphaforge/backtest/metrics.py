"""Performance analytics for a return stream.

Every statistic here is computed from the *realised* return series, so it
includes whatever the strategy actually paid in costs, slippage and impact.
Nothing is annualised by assumption alone: the annualisation factor is inferred
from the index, and it is reported alongside the numbers that depend on it.

Definitions worth stating because implementations differ
--------------------------------------------------------
``sharpe``
    ``(mean - rf/periods) / std * sqrt(periods)`` - excess return over the
    period's risk-free accrual, not over zero.
``sortino``
    same numerator, denominator = downside deviation computed against a
    zero target with **all** observations in the denominator (Sortino-Satchell
    convention), so a flat series does not get an infinite ratio by accident.
``max_drawdown``
    worst peak-to-trough of the compounded curve, in negative terms.
``alpha``
    Jensen's alpha from an OLS of excess strategy returns on excess benchmark
    returns, annualised from the regression intercept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger
from alphaforge.utils.math_utils import (
    TRADING_DAYS_PER_YEAR,
    compound,
    drawdown_series,
    infer_periods_per_year,
    safe_div,
)

log = get_logger("backtest.metrics")


@dataclass
class MetricsConfig:
    risk_free_rate: float = 0.0
    periods_per_year: float | None = None  # inferred from the index when None
    var_level: float = 0.95
    rolling_window: int = 252

    @classmethod
    def from_dict(cls, cfg: dict | None) -> MetricsConfig:
        cfg = cfg or {}
        ppy = cfg.get("periods_per_year")
        return cls(
            risk_free_rate=float(cfg.get("risk_free_rate", 0.0)),
            periods_per_year=None if ppy in (None, "null") else float(ppy),
            var_level=float(cfg.get("var_level", 0.95)),
            rolling_window=int(cfg.get("rolling_window", 252)),
        )


# --------------------------------------------------------------------------
def performance_stats(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    config: MetricsConfig | dict | None = None,
    turnover: pd.Series | None = None,
    cost_drag: float | None = None,
) -> dict:
    """Full performance summary for one return stream."""
    cfg = config if isinstance(config, MetricsConfig) else MetricsConfig.from_dict(config or {})
    r = returns.dropna().astype(float)
    if r.empty:
        return {"error": "empty return series"}

    ppy = cfg.periods_per_year or infer_periods_per_year(pd.DatetimeIndex(r.index))
    rf_period = (1.0 + cfg.risk_free_rate) ** (1.0 / ppy) - 1.0
    excess = r - rf_period

    curve = compound(r)
    total_return = float(curve.iloc[-1] - 1.0)
    years = safe_div(len(r), ppy, float("nan"))
    cagr = float(curve.iloc[-1] ** (1.0 / years) - 1.0) if years and years > 0 else float("nan")

    vol = float(r.std(ddof=1) * np.sqrt(ppy))
    downside = excess.clip(upper=0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(ppy))
    sharpe = safe_div(float(excess.mean()) * ppy, vol)
    sortino = safe_div(float(excess.mean()) * ppy, downside_dev)

    mdd, mdd_date = _max_drawdown(r)
    calmar = safe_div(cagr, abs(mdd)) if mdd else float("nan")

    var_level = float(np.quantile(r, 1.0 - cfg.var_level))
    tail = r[r <= var_level]
    cvar = float(tail.mean()) if len(tail) else float("nan")

    stats = {
        "start": str(pd.Timestamp(r.index[0]).date()),
        "end": str(pd.Timestamp(r.index[-1]).date()),
        "n_periods": int(len(r)),
        "years": float(years) if years == years else float("nan"),
        "periods_per_year": float(ppy),
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": vol,
        "ann_downside_dev": downside_dev,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": float(mdd),
        "max_drawdown_date": str(pd.Timestamp(mdd_date).date()) if mdd_date is not None else None,
        "max_drawdown_duration_days": int(_drawdown_duration(r)),
        "var_95": var_level,
        "cvar_95": cvar,
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "best_period": float(r.max()),
        "worst_period": float(r.min()),
        "positive_period_share": float((r > 0).mean()),
        "avg_period_return": float(r.mean()),
        "hit_rate": float((r > 0).mean()),
        "avg_turnover": float(turnover.mean())
        if turnover is not None and len(turnover)
        else float("nan"),
        "cost_drag_ann": float(cost_drag) if cost_drag is not None else float("nan"),
    }

    if benchmark is not None:
        stats.update(_relative_stats(r, benchmark.dropna().astype(float), rf_period, ppy))
    return stats


def _relative_stats(r: pd.Series, bench: pd.Series, rf_period: float, ppy: float) -> dict:
    """Beta, alpha, tracking error, information ratio and capture ratios."""
    joined = pd.concat([r.rename("strategy"), bench.rename("benchmark")], axis=1).dropna()
    if len(joined) < 3:
        return {}
    s = joined["strategy"]
    b = joined["benchmark"]
    se, be = s - rf_period, b - rf_period

    cov = float(np.cov(se, be, ddof=1)[0, 1])
    var_b = float(np.var(be, ddof=1))
    beta = safe_div(cov, var_b)
    alpha_period = float(se.mean() - beta * be.mean())

    active = s - b
    te = float(active.std(ddof=1) * np.sqrt(ppy))
    strat_curve = compound(s)
    bench_curve = compound(b)
    bench_cagr = float(bench_curve.iloc[-1] ** (ppy / max(len(b), 1)) - 1.0)

    up = b > 0
    down = b < 0
    up_capture = safe_div(float(s[up].mean()), float(b[up].mean())) if up.any() else float("nan")
    down_capture = (
        safe_div(float(s[down].mean()), float(b[down].mean())) if down.any() else float("nan")
    )

    return {
        "benchmark_return": float(bench_curve.iloc[-1] - 1.0),
        "benchmark_cagr": bench_cagr,
        "benchmark_vol": float(b.std(ddof=1) * np.sqrt(ppy)),
        # Active return = terminal wealth difference, i.e. the arithmetic gap
        # between the two compounded curves over the full period. NOTE: it is
        # NOT compound(s - b) - 1. `s - b` is an arithmetic return *spread*, not
        # a return, so compounding it is meaningless and numerically explosive:
        # whenever s - b < -1 the factor 1 + (s - b) turns negative with
        # magnitude > 1, and a few thousand sessions of cumprod overflow float64.
        "active_return": float(strat_curve.iloc[-1] - bench_curve.iloc[-1]),
        "beta": beta,
        "alpha_ann": alpha_period * ppy,
        "correlation": float(s.corr(b)),
        "r_squared": float(s.corr(b) ** 2) if s.std() and b.std() else float("nan"),
        "tracking_error": te,
        "information_ratio": safe_div(float(active.mean()) * ppy, te),
        "up_capture": up_capture,
        "down_capture": down_capture,
        "treynor": safe_div(float(se.mean()) * ppy, beta),
    }


# --------------------------------------------------------------------------
def _max_drawdown(returns: pd.Series) -> tuple[float, pd.Timestamp | None]:
    dd = drawdown_series(returns)
    if dd.empty:
        return 0.0, None
    return float(dd.min()), pd.Timestamp(dd.idxmin())


def _drawdown_duration(returns: pd.Series) -> int:
    """Longest stretch (in periods) spent below a previous peak."""
    dd = drawdown_series(returns)
    longest = current = 0
    for value in dd.to_numpy(dtype=float):
        if value < -1e-12:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def gross_returns_from_net(
    returns: pd.Series,
    costs: pd.Series,
    equity: pd.Series,
    initial_capital: float,
) -> pd.Series:
    """Recover the pre-cost (gross) return series from a net series.

    Transaction costs are deducted from NAV at the end of the session, so the
    net return for day ``d`` is ``(nav_d - cost_d) / prev_nav_d - 1``.  Adding
    the fractional drag ``cost_d / prev_nav_d`` back recovers the gross return
    (the return the strategy would have earned had it paid no costs).  This makes
    the *cost drag* explicit and reconciles gross vs net Sharpe/CAGR exactly,
    without re-running the backtest.

    Parameters
    ----------
    returns : net daily return series (the realised, cost-inclusive one).
    costs   : per-day currency cost (often sparser than ``returns``; reindexed).
    equity  : end-of-day net NAV curve, indexed like ``returns``.
    initial_capital : NAV before the first return is earned.
    """
    idx = returns.index
    prev_nav = equity.reindex(idx).shift(1)
    prev_nav.iloc[0] = float(initial_capital)
    cost = costs.reindex(idx).fillna(0.0)
    # Guard against a zero/NaN starting NAV (e.g. first day with no prior book).
    addback = (cost / prev_nav.replace(0.0, np.nan)).fillna(0.0)
    return (returns + addback).reindex(idx)


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Year x month table of compounded returns (the classic heat-map input)."""
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()
    grouped = r.groupby([r.index.year, r.index.month]).apply(
        lambda s: float(compound(s).iloc[-1] - 1.0)
    )
    table = grouped.unstack()
    table.columns = [pd.Timestamp(2000, int(m), 1).strftime("%b") for m in table.columns]
    table.index.name = "Year"
    return table


def yearly_returns(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    return r.groupby(r.index.year).apply(lambda s: float(compound(s).iloc[-1] - 1.0))


def rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    benchmark: pd.Series | None = None,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Rolling return, volatility, Sharpe and drawdown."""
    r = returns.dropna()
    out = pd.DataFrame(index=r.index)
    out["rolling_return"] = compound(r).pct_change(window)
    out["rolling_vol"] = r.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    out["rolling_sharpe"] = (
        r.rolling(window).mean() * periods_per_year / out["rolling_vol"].replace(0, np.nan)
    )
    out["drawdown"] = drawdown_series(r)
    if benchmark is not None:
        b = benchmark.reindex(r.index).fillna(0.0)
        out["rolling_beta"] = r.rolling(window).cov(b) / b.rolling(window).var().replace(0, np.nan)
    return out


def drawdown_table(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """The ``top`` deepest drawdown episodes with their recovery profile."""
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()
    curve = compound(r)
    peak = curve.cummax()
    dd = curve / peak - 1.0

    episodes = []
    in_dd = False
    start = None
    for ts, value in dd.items():
        if value < -1e-12 and not in_dd:
            in_dd, start = True, ts
        elif value >= -1e-12 and in_dd:
            in_dd = False
            episodes.append((start, ts))
    if in_dd:
        episodes.append((start, dd.index[-1]))

    rows = []
    for start, end in episodes:
        window = dd.loc[start:end]
        trough = window.idxmin()
        recovered = end
        rows.append(
            {
                "start": str(pd.Timestamp(start).date()),
                "trough": str(pd.Timestamp(trough).date()),
                "end": str(pd.Timestamp(recovered).date()),
                "depth": float(window.min()),
                "length_days": int(len(window)),
                "to_trough_days": int(dd.index.get_loc(trough) - dd.index.get_loc(start)),
                "recovery_days": int(dd.index.get_loc(recovered) - dd.index.get_loc(trough)),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("depth").head(top).reset_index(drop=True)


def summarise(metrics: dict) -> pd.DataFrame:
    """Key/value frame for rendering in a report."""
    return pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})


__all__ = [
    "MetricsConfig",
    "performance_stats",
    "gross_returns_from_net",
    "monthly_returns",
    "yearly_returns",
    "rolling_metrics",
    "drawdown_table",
    "summarise",
]

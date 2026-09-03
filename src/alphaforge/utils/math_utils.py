"""Deterministic numerical helpers used across the research pipeline.

Every helper is written to be *cross-section aware*: operations are applied
per rebalance date so that information from one date can never bleed into
another.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------
# Period / annualisation helpers
# --------------------------------------------------------------------------
def infer_periods_per_year(index: pd.DatetimeIndex | pd.Index) -> float:
    """Infer annualisation factor from an index of observation dates.

    The conversion goes through pandas' timedelta machinery rather than raw
    ``int64`` ticks: pandas >= 3 stores datetimes in microseconds by default, so
    a hardcoded nanosecond divisor silently inflates the factor by 1000x and
    turns a 12% vol into a 380% one.
    """
    if len(index) < 3:
        return TRADING_DAYS_PER_YEAR
    idx = pd.DatetimeIndex(index)
    gaps = pd.Series(idx).diff().dropna()
    if gaps.empty:
        return TRADING_DAYS_PER_YEAR
    median_gap_days = float(gaps.median().total_seconds()) / 86_400.0
    if median_gap_days <= 0:
        return TRADING_DAYS_PER_YEAR
    # Daily observations (business or calendar) cannot be distinguished from the
    # median gap alone, and every day-scale quant return is conventionally annualised
    # with 252 trading days - using 365 over-states vol and CAGR by ~20%.
    if 0.8 <= median_gap_days <= 1.6:
        return float(TRADING_DAYS_PER_YEAR)
    return float(365.0 / median_gap_days)


def annualize_return(mean_period_return: float, periods_per_year: float) -> float:
    return (1.0 + mean_period_return) ** periods_per_year - 1.0


def annualize_vol(std_period_return: float, periods_per_year: float) -> float:
    return float(std_period_return) * float(np.sqrt(periods_per_year))


# --------------------------------------------------------------------------
# Cross-sectional transforms
# --------------------------------------------------------------------------
def winsorize(series: pd.Series, lower: float = 0.01, upper: float | None = None) -> pd.Series:
    """Two-sided winsorization at the given quantiles."""
    upper = lower if upper is None else upper
    if series.dropna().empty:
        return series
    lo = series.quantile(lower)
    hi = series.quantile(1.0 - upper)
    return series.clip(lower=lo, upper=hi)


def zscore(df: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """Cross-sectional (axis=1) or time-series (axis=0) z-score."""
    mean = df.mean(axis=axis)
    std = df.std(axis=axis, ddof=1).replace(0.0, np.nan)
    scaled = df.sub(mean, axis=0 if axis == 1 else 1).div(std, axis=0 if axis == 1 else 1)
    return scaled


def rank_transform(df: pd.DataFrame, axis: int = 1, pct: bool = True) -> pd.DataFrame:
    return df.rank(axis=axis, pct=pct, na_option="keep")


def demean(df: pd.DataFrame, group: pd.Series | None = None, axis: int = 1) -> pd.DataFrame:
    """Cross-sectional demeaning, optionally within groups (e.g. industry).

    With ``group`` the mean is taken *inside* each group only, so an industry
    label never leaks its level into another industry's residual.

    Parameters
    ----------
    group:
        ``axis=1`` -> maps column labels to group ids (e.g. symbol -> sector);
        ``axis=0`` -> maps row labels to group ids (e.g. date -> regime).
    """
    if group is None:
        return df.sub(df.mean(axis=axis), axis=0 if axis == 1 else 1)

    if axis == 1:
        g = group.reindex(df.columns)
        out = df.copy()
        for _, cols in g.groupby(g).groups.items():
            cols = [c for c in cols if c in df.columns]
            block = df[cols]
            out[cols] = block.sub(block.mean(axis=1), axis=0)
        return out

    g = group.reindex(df.index)
    return df.groupby(g).transform(lambda x: x - x.mean())


def neutralize(
    factor: pd.Series,
    exposures: pd.DataFrame,
    add_constant: bool = True,
) -> pd.Series:
    """Residualise ``factor`` against ``exposures`` via OLS on a single date.

    Parameters
    ----------
    factor:
        Cross-section of raw factor values indexed by asset id.
    exposures:
        DataFrame of style / dummy exposures indexed by the same asset ids.
    add_constant:
        Include an intercept (standard for industry neutralisation).

    Returns
    -------
    pd.Series of residuals with the same index as ``factor``.
    """
    data = pd.concat([factor.rename("__f__"), exposures], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty or len(data) <= exposures.shape[1] + int(add_constant):
        return pd.Series(np.nan, index=factor.index)

    y = data["__f__"].to_numpy(dtype=float)
    x = data.drop(columns="__f__").to_numpy(dtype=float)
    if add_constant:
        x = np.column_stack([np.ones(len(x)), x])

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    out = pd.Series(np.nan, index=factor.index, dtype=float)
    out.loc[data.index] = resid
    return out


# --------------------------------------------------------------------------
# Return helpers
# --------------------------------------------------------------------------
def to_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Period returns from a price panel (dates x assets)."""
    if method == "log":
        return np.log(prices / prices.shift(1))
    return prices.pct_change(fill_method=None)


def forward_returns(
    prices: pd.DataFrame, horizon: int = 21, method: str = "simple"
) -> pd.DataFrame:
    """Forward-looking ``horizon``-period return, aligned to the *decision* date.

    The value at row ``t`` is the return realised from ``t`` to ``t + horizon``
    and is therefore only knowable after ``t``. It must be used as a *label*,
    never as a feature.
    """
    if method == "log":
        fwd = np.log(prices.shift(-horizon) / prices)
    else:
        fwd = prices.shift(-horizon) / prices - 1.0
    return fwd


def compound(returns: pd.Series) -> pd.Series:
    """Cumulative wealth curve for a **return** series (``1.0`` = initial capital).

    Accumulates in log space: ``exp(cumsum(log1p(r)))``. This is mathematically
    identical to ``(1 + r).cumprod()`` for ``r > -1`` but numerically stable over
    long horizons — a naive ``cumprod`` silently leaves the float64 range
    (overflowing to ``inf`` or underflowing to ``0``) on multi-thousand-session
    backtests, which then poisons every downstream statistic (CAGR, drawdown,
    Sharpe) with ``nan`` instead of a number.

    The input must be *returns*, not return spreads. See
    :func:`alphaforge.backtest.metrics._relative_stats` for the trap this guards.
    """
    r = returns.fillna(0.0).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_growth = np.log1p(r)
    # r <= -1 means the position is wiped out; clamp to zero wealth rather than
    # propagating inf/nan through the whole curve.
    log_growth = np.where(np.isfinite(log_growth), log_growth, -np.inf)
    return pd.Series(np.exp(np.cumsum(log_growth)), index=returns.index, name=returns.name)


def max_drawdown(returns: pd.Series) -> tuple[float, pd.Timestamp | None]:
    curve = compound(returns)
    running_max = curve.cummax()
    dd = curve / running_max - 1.0
    if dd.empty:
        return 0.0, None
    trough = dd.idxmin()
    return float(dd.min()), trough


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = compound(returns)
    return curve / curve.cummax() - 1.0


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank (Spearman) correlation between two aligned cross-sections."""
    df = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3:
        return float("nan")
    return float(df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank()))


def pearson(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def safe_div(num: float, den: float, default: float = float("nan")) -> float:
    if den is None or abs(den) < 1e-12 or np.isnan(den):
        return default
    return float(num) / float(den)


def expanding_window_index(n: int, min_obs: int) -> Iterable[tuple[np.ndarray, int]]:
    """Yield ``(train_idx, test_position)`` pairs for expanding-window CV."""
    for i in range(min_obs, n):
        yield np.arange(0, i, dtype=int), i


def first_valid_columns(df: pd.DataFrame, min_periods: int) -> Sequence[str]:
    counts = df.notna().sum()
    return list(counts[counts >= min_periods].index)


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "MONTHS_PER_YEAR",
    "infer_periods_per_year",
    "annualize_return",
    "annualize_vol",
    "winsorize",
    "zscore",
    "rank_transform",
    "demean",
    "neutralize",
    "to_returns",
    "forward_returns",
    "compound",
    "max_drawdown",
    "drawdown_series",
    "spearman",
    "pearson",
    "safe_div",
    "expanding_window_index",
    "first_valid_columns",
]

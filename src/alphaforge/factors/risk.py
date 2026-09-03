"""Risk / volatility factors.

All risk measures are computed from **trailing** windows only. Realised
volatility is the canonical low-volatility-anomaly exposure; idiosyncratic
volatility strips the market factor out first so the signal is not just a
re-labelled beta.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from alphaforge.factors.base import FactorContext, FactorSpec, register
from alphaforge.utils.logging import get_logger

log = get_logger("factors.risk")

_ANN = np.sqrt(252.0)


def _market_return(ctx: FactorContext) -> pd.Series:
    rets = ctx.returns.where(ctx.panel.universe)
    if ctx.panel.benchmark is not None:
        bench = ctx.panel.benchmark.reindex(rets.index).pct_change(fill_method=None)
        if bench.notna().sum() > 100:
            return bench
    return rets.mean(axis=1)


@register(
    FactorSpec(
        name="volatility_60d",
        category="risk",
        direction=-1,
        description="Annualised 60-day realised volatility. Direction -1 (low-vol anomaly).",
        data_requirement="price",
    )
)
def volatility_60d(ctx: FactorContext) -> pd.DataFrame:
    return ctx.returns.rolling(60, min_periods=30).std() * _ANN


@register(
    FactorSpec(
        name="volatility_252d",
        category="risk",
        direction=-1,
        description="Annualised 1-year realised volatility.",
        data_requirement="price",
    )
)
def volatility_252d(ctx: FactorContext) -> pd.DataFrame:
    return ctx.returns.rolling(252, min_periods=126).std() * _ANN


@register(
    FactorSpec(
        name="downside_volatility",
        category="risk",
        direction=-1,
        description="Annualised downside deviation over 126 days (semi-deviation below zero).",
        data_requirement="price",
    )
)
def downside_volatility(ctx: FactorContext) -> pd.DataFrame:
    r = ctx.returns
    neg = r.clip(upper=0.0)
    dd = np.sqrt((neg**2).rolling(126, min_periods=60).mean())
    return dd * _ANN


@register(
    FactorSpec(
        name="beta_252d",
        category="risk",
        direction=-1,
        description="Rolling 252-day market beta (low beta anomaly).",
        data_requirement="price",
    )
)
def beta_252d(ctx: FactorContext) -> pd.DataFrame:
    rets = ctx.returns.where(ctx.panel.universe)
    mkt = _market_return(ctx)
    cov = rets.rolling(252, min_periods=126).cov(mkt)
    var = mkt.rolling(252, min_periods=126).var()
    return cov.div(var.replace(0.0, np.nan), axis=0)


@register(
    FactorSpec(
        name="idiosyncratic_volatility",
        category="risk",
        direction=-1,
        description=(
            "Annualised standard deviation of the residual from a rolling 252-day "
            "market-model regression (Ang, Hodrick, Xing & Zhang)."
        ),
        data_requirement="price",
    )
)
def idiosyncratic_volatility(ctx: FactorContext) -> pd.DataFrame:
    rets = ctx.returns.where(ctx.panel.universe)
    mkt = _market_return(ctx)
    w = 252
    cov = rets.rolling(w, min_periods=126).cov(mkt)
    var = mkt.rolling(w, min_periods=126).var()
    beta = cov.div(var.replace(0.0, np.nan), axis=0)
    resid = rets - beta.mul(mkt, axis=0)
    return resid.rolling(w, min_periods=126).std() * _ANN


@register(
    FactorSpec(
        name="max_drawdown_252d",
        category="risk",
        direction=-1,
        description="Worst peak-to-trough decline over a rolling 252-day window.",
        data_requirement="price",
    )
)
def max_drawdown_252d(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close.where(ctx.panel.universe)
    out = close.copy() * np.nan
    values = close.to_numpy(dtype=float)
    n = values.shape[0]
    res = np.full(values.shape, np.nan)
    for i in range(252, n):
        window = values[i - 252 : i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            peak = np.nanmax(
                np.fmax.accumulate(np.where(np.isnan(window), -np.inf, window)), axis=0
            )
            trough = np.nanmin(window, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            res[i] = trough / np.where(peak > 0, peak, np.nan) - 1.0
    out[:] = res
    return out


@register(
    FactorSpec(
        name="return_skew_126d",
        category="risk",
        direction=-1,
        description="Rolling 126-day return skewness; negatively skewed names are riskier.",
        data_requirement="price",
    )
)
def return_skew_126d(ctx: FactorContext) -> pd.DataFrame:
    return ctx.returns.rolling(126, min_periods=60).skew()


__all__ = [
    "volatility_60d",
    "volatility_252d",
    "downside_volatility",
    "beta_252d",
    "idiosyncratic_volatility",
    "max_drawdown_252d",
    "return_skew_126d",
]

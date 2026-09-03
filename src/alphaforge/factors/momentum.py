"""Momentum and short-horizon reversal factors.

Returns are computed from **adjusted** prices so that splits and dividends do
not masquerade as momentum.  The classic 12-1 specification deliberately skips
the most recent month, which is contaminated by short-term reversal and the
bid-ask bounce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.factors.base import FactorContext, FactorSpec, register
from alphaforge.utils.logging import get_logger

log = get_logger("factors.momentum")


def _momentum(close: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Total return from ``t-lookback-skip`` to ``t-skip``."""
    if skip:
        return close.shift(skip) / close.shift(lookback + skip) - 1.0
    return close / close.shift(lookback) - 1.0


@register(
    FactorSpec(
        name="mom_20d",
        category="momentum",
        direction=1,
        description="1-month (20 trading day) price momentum.",
        data_requirement="price",
    )
)
def mom_20d(ctx: FactorContext) -> pd.DataFrame:
    return _momentum(ctx.close, 20)


@register(
    FactorSpec(
        name="mom_60d",
        category="momentum",
        direction=1,
        description="3-month (60 trading day) price momentum.",
    )
)
def mom_60d(ctx: FactorContext) -> pd.DataFrame:
    return _momentum(ctx.close, 60)


@register(
    FactorSpec(
        name="mom_120d",
        category="momentum",
        direction=1,
        description="6-month (120 trading day) price momentum.",
    )
)
def mom_120d(ctx: FactorContext) -> pd.DataFrame:
    return _momentum(ctx.close, 120)


@register(
    FactorSpec(
        name="mom_12_1",
        category="momentum",
        direction=1,
        description="Classic 12-1 momentum: 250-day return skipping the most recent 21 days.",
    )
)
def mom_12_1(ctx: FactorContext) -> pd.DataFrame:
    return _momentum(ctx.close, lookback=250, skip=21)


@register(
    FactorSpec(
        name="mom_6_1",
        category="momentum",
        direction=1,
        description="6-1 momentum: 126-day return skipping the most recent 21 days.",
    )
)
def mom_6_1(ctx: FactorContext) -> pd.DataFrame:
    return _momentum(ctx.close, lookback=126, skip=21)


@register(
    FactorSpec(
        name="rev_5d",
        category="reversal",
        direction=-1,
        description="5-day short-term reversal (higher past return => lower expected return).",
    )
)
def rev_5d(ctx: FactorContext) -> pd.DataFrame:
    return _momentum(ctx.close, 5)


@register(
    FactorSpec(
        name="rev_21d",
        category="reversal",
        direction=-1,
        description="1-month reversal, the classic short-horizon contrarian signal.",
    )
)
def rev_21d(ctx: FactorContext) -> pd.DataFrame:
    return _momentum(ctx.close, 21)


@register(
    FactorSpec(
        name="residual_momentum",
        category="momentum",
        direction=1,
        description=(
            "Residual momentum (Blitz, Huij & Martens): the intercept from regressing a "
            "12-1 window of daily stock returns on the contemporaneous equal-weighted "
            "market return. Isolates stock-specific trend from market beta."
        ),
    )
)
def residual_momentum(ctx: FactorContext) -> pd.DataFrame:
    """Vectorised Fama-French style residual momentum.

    ``alpha_i = mean(r_i) - beta_i * mean(r_mkt)`` over a 250-day window, with
    ``beta_i = cov(r_i, r_mkt) / var(r_mkt)``.  Computed with rolling moments
    rather than 400k separate OLS fits (identical estimates, ~1000x faster).
    """
    rets = ctx.returns.where(ctx.panel.universe)
    mkt = rets.mean(axis=1)
    window, skip = 250, 21

    cov = rets.rolling(window, min_periods=max(window // 2, 60)).cov(mkt)
    var = mkt.rolling(window, min_periods=max(window // 2, 60)).var()
    beta = cov.div(var.replace(0.0, np.nan), axis=0)
    mean_r = rets.rolling(window, min_periods=max(window // 2, 60)).mean()
    mean_m = mkt.rolling(window, min_periods=max(window // 2, 60)).mean()
    alpha = mean_r.sub(beta.mul(mean_m, axis=0))
    return (alpha * window).shift(skip)


@register(
    FactorSpec(
        name="industry_momentum",
        category="momentum",
        direction=1,
        description="60-day equal-weighted industry momentum mapped back onto members.",
    )
)
def industry_momentum(ctx: FactorContext) -> pd.DataFrame:
    rets = ctx.returns.where(ctx.panel.universe)
    mapping = ctx.panel.industry.ffill().iloc[-1]
    ind_rets = rets.T.groupby(mapping).transform("mean").T
    ind_mom = (1.0 + ind_rets.fillna(0.0)).rolling(60).apply(np.prod, raw=True) - 1.0
    return ind_mom.shift(1)


__all__ = [
    "mom_20d",
    "mom_60d",
    "mom_120d",
    "mom_12_1",
    "mom_6_1",
    "rev_5d",
    "rev_21d",
    "residual_momentum",
    "industry_momentum",
]

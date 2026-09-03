"""Liquidity and size factors.

Liquidity is the practical constraint that decides whether a backtest is
investable: an illiquid signal with a beautiful backtest is worthless. These
factors are used both as alphas (illiquidity premium) and as screens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.factors.base import FactorContext, FactorSpec, register
from alphaforge.utils.logging import get_logger

log = get_logger("factors.liquidity")


@register(
    FactorSpec(
        name="log_market_cap",
        category="size",
        direction=-1,
        description="Natural log of market capitalisation. Direction -1 (small-cap premium).",
        data_requirement="price:market_cap",
    )
)
def log_market_cap(ctx: FactorContext) -> pd.DataFrame:
    return np.log(ctx.market_cap.replace(0.0, np.nan))


@register(
    FactorSpec(
        name="log_price",
        category="size",
        direction=-1,
        description="Natural log of the adjusted price (a low-price / retail tilt proxy).",
        data_requirement="price",
    )
)
def log_price(ctx: FactorContext) -> pd.DataFrame:
    return np.log(ctx.close.replace(0.0, np.nan))


@register(
    FactorSpec(
        name="adv_21d",
        category="liquidity",
        direction=1,
        description="21-day average dollar volume (ADV).",
        data_requirement="price:volume",
    )
)
def adv_21d(ctx: FactorContext) -> pd.DataFrame:
    return ctx.panel.dollar_volume.rolling(21, min_periods=10).mean()


@register(
    FactorSpec(
        name="log_adv_21d",
        category="liquidity",
        direction=1,
        description="Log 21-day ADV - the standard liquidity control in cross-sectional studies.",
        data_requirement="price:volume",
    )
)
def log_adv_21d(ctx: FactorContext) -> pd.DataFrame:
    adv = ctx.panel.dollar_volume.rolling(21, min_periods=10).mean()
    return np.log(adv.replace(0.0, np.nan))


@register(
    FactorSpec(
        name="turnover_21d",
        category="liquidity",
        direction=-1,
        description="21-day average share volume / shares outstanding.",
        data_requirement="price:volume,shares_outstanding",
    )
)
def turnover_21d(ctx: FactorContext) -> pd.DataFrame:
    shares = ctx.panel.market_cap / ctx.close.replace(0.0, np.nan)
    tv = ctx.panel.volume.rolling(21, min_periods=10).mean() / shares.replace(0.0, np.nan)
    return tv


@register(
    FactorSpec(
        name="amihud_illiquidity",
        category="liquidity",
        direction=1,
        description=(
            "Amihud (2002) illiquidity: mean of |return| / dollar volume over 21 days, "
            "scaled by 1e6. Higher = less liquid."
        ),
        data_requirement="price:volume",
    )
)
def amihud_illiquidity(ctx: FactorContext) -> pd.DataFrame:
    daily_illi = ctx.returns.abs() / ctx.panel.dollar_volume.replace(0.0, np.nan)
    return daily_illi.rolling(21, min_periods=10).mean() * 1e6


@register(
    FactorSpec(
        name="dollar_volume_ratio",
        category="liquidity",
        direction=1,
        description="21-day ADV / 252-day ADV - recent liquidity expansion or contraction.",
        data_requirement="price:volume",
    )
)
def dollar_volume_ratio(ctx: FactorContext) -> pd.DataFrame:
    dv = ctx.panel.dollar_volume
    short = dv.rolling(21, min_periods=10).mean()
    long = dv.rolling(252, min_periods=126).mean()
    return short / long.replace(0.0, np.nan)


@register(
    FactorSpec(
        name="zero_trading_days",
        category="liquidity",
        direction=-1,
        description="Fraction of zero-volume days over the last 21 sessions (Lesmond-style proxy).",
        data_requirement="price:volume",
    )
)
def zero_trading_days(ctx: FactorContext) -> pd.DataFrame:
    vol = ctx.panel.volume
    return (vol.fillna(0.0) <= 0).rolling(21, min_periods=10).mean()


__all__ = [
    "log_market_cap",
    "log_price",
    "adv_21d",
    "log_adv_21d",
    "turnover_21d",
    "amihud_illiquidity",
    "dollar_volume_ratio",
    "zero_trading_days",
]

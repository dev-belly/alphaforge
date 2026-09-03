"""Quality / profitability factors (point-in-time).

Quality is the factor group most sensitive to look-ahead bias: accounting
figures are revised, restated and (critically) published with a lag. All of the
signals below are built from the most recent *publicly released* statement only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.factors.base import FactorContext, FactorSpec, register
from alphaforge.utils.logging import get_logger

log = get_logger("factors.quality")


def _ratio(ctx: FactorContext, name: str) -> pd.DataFrame:
    return ctx.require_fundamentals().ratio(name)


@register(
    FactorSpec(
        name="roe",
        category="quality",
        direction=1,
        description="Return on equity: net income / book equity.",
        requires_fundamentals=True,
        data_requirement="fundamental:net_income,total_equity",
    )
)
def roe(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "roe")


@register(
    FactorSpec(
        name="roa",
        category="quality",
        direction=1,
        description="Return on assets: net income / total assets.",
        requires_fundamentals=True,
        data_requirement="fundamental:net_income,total_assets",
    )
)
def roa(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "roa")


@register(
    FactorSpec(
        name="gross_profitability",
        category="quality",
        direction=1,
        description="Novy-Marx gross profitability: gross profit / total assets.",
        requires_fundamentals=True,
        data_requirement="fundamental:gross_profit,total_assets",
    )
)
def gross_profitability(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "gross_profitability")


@register(
    FactorSpec(
        name="asset_turnover",
        category="quality",
        direction=1,
        description="Revenue / total assets - operating efficiency.",
        requires_fundamentals=True,
        data_requirement="fundamental:revenue,total_assets",
    )
)
def asset_turnover(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "asset_turnover")


@register(
    FactorSpec(
        name="earnings_quality",
        category="quality",
        direction=1,
        description="(Operating cash flow - net income) / total assets. Higher = less accrual-driven.",
        requires_fundamentals=True,
        data_requirement="fundamental:operating_cashflow,net_income,total_assets",
    )
)
def earnings_quality(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "earnings_quality")


@register(
    FactorSpec(
        name="accruals",
        category="quality",
        direction=-1,
        description="(Net income - operating cash flow) / total assets. High accruals are bearish.",
        requires_fundamentals=True,
        data_requirement="fundamental:net_income,operating_cashflow,total_assets",
    )
)
def accruals(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "accruals")


@register(
    FactorSpec(
        name="gross_margin",
        category="quality",
        direction=1,
        description="Gross profit / revenue.",
        requires_fundamentals=True,
        data_requirement="fundamental:gross_profit,revenue",
    )
)
def gross_margin(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "gross_margin")


@register(
    FactorSpec(
        name="low_leverage",
        category="quality",
        direction=-1,
        description="Total debt / total assets, signed so that higher = safer balance sheet.",
        requires_fundamentals=True,
        data_requirement="fundamental:total_debt,total_assets",
    )
)
def low_leverage(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "leverage")


@register(
    FactorSpec(
        name="quality_composite",
        category="quality",
        direction=1,
        description="Equal-weighted z-score of ROE, ROA, gross profitability and earnings quality.",
        requires_fundamentals=True,
        data_requirement="fundamental:multiple",
    )
)
def quality_composite(ctx: FactorContext) -> pd.DataFrame:
    parts = []
    for key in ("roe", "roa", "gross_profitability", "earnings_quality"):
        try:
            parts.append(_ratio(ctx, key))
        except Exception:  # noqa: BLE001
            continue
    if not parts:
        raise RuntimeError("No quality inputs available")
    zs = [
        p.sub(p.mean(axis=1), axis=0).div(p.std(axis=1).replace(0, np.nan), axis=0) for p in parts
    ]
    return pd.concat(zs).groupby(level=0).mean()


__all__ = [
    "roe",
    "roa",
    "gross_profitability",
    "asset_turnover",
    "earnings_quality",
    "accruals",
    "gross_margin",
    "low_leverage",
    "quality_composite",
]

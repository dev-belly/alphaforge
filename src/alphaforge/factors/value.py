"""Value factors built from **point-in-time** fundamentals.

Every ratio uses market capitalisation observed on the signal date and a
statement whose ``report_date`` is on or before that date.  No fiscal-period
joins anywhere - see :mod:`alphaforge.features.fundamentals`.

If the configured provider supplies no fundamentals the registry raises
:class:`FactorUnavailableError` and the factor is reported as unavailable
rather than being silently faked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.factors.base import FactorContext, FactorSpec, register
from alphaforge.utils.logging import get_logger

log = get_logger("factors.value")


def _ratio(ctx: FactorContext, name: str) -> pd.DataFrame:
    fund = ctx.require_fundamentals()
    return fund.ratio(name)


@register(
    FactorSpec(
        name="earnings_yield",
        category="value",
        direction=1,
        description="Net income / market capitalisation (inverse P/E).",
        requires_fundamentals=True,
        data_requirement="fundamental:net_income,market_cap",
    )
)
def earnings_yield(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "earnings_yield")


@register(
    FactorSpec(
        name="book_to_price",
        category="value",
        direction=1,
        description="Book equity / market capitalisation (inverse P/B).",
        requires_fundamentals=True,
        data_requirement="fundamental:total_equity,market_cap",
    )
)
def book_to_price(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "book_to_price")


@register(
    FactorSpec(
        name="sales_to_price",
        category="value",
        direction=1,
        description="Revenue / market capitalisation (inverse P/S).",
        requires_fundamentals=True,
        data_requirement="fundamental:revenue,market_cap",
    )
)
def sales_to_price(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "sales_to_price")


@register(
    FactorSpec(
        name="fcf_yield",
        category="value",
        direction=1,
        description="(Operating cash flow - capex) / market capitalisation.",
        requires_fundamentals=True,
        data_requirement="fundamental:operating_cashflow,capex,market_cap",
    )
)
def fcf_yield(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "fcf_yield")


@register(
    FactorSpec(
        name="ebit_to_ev",
        category="value",
        direction=1,
        description="EBIT / enterprise value (market cap + total debt).",
        requires_fundamentals=True,
        data_requirement="fundamental:ebit,total_debt,market_cap",
    )
)
def ebit_to_ev(ctx: FactorContext) -> pd.DataFrame:
    return _ratio(ctx, "ebit_to_ev")


@register(
    FactorSpec(
        name="pe_ratio",
        category="value",
        direction=-1,
        description="Price / earnings. Stored with direction -1 so higher rank = cheaper.",
        requires_fundamentals=True,
        data_requirement="fundamental:net_income,market_cap",
    )
)
def pe_ratio(ctx: FactorContext) -> pd.DataFrame:
    ey = _ratio(ctx, "earnings_yield")
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / ey.replace(0.0, np.nan)


@register(
    FactorSpec(
        name="pb_ratio",
        category="value",
        direction=-1,
        description="Price / book. Stored with direction -1 so higher rank = cheaper.",
        requires_fundamentals=True,
        data_requirement="fundamental:total_equity,market_cap",
    )
)
def pb_ratio(ctx: FactorContext) -> pd.DataFrame:
    bp = _ratio(ctx, "book_to_price")
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / bp.replace(0.0, np.nan)


@register(
    FactorSpec(
        name="ps_ratio",
        category="value",
        direction=-1,
        description="Price / sales. Stored with direction -1 so higher rank = cheaper.",
        requires_fundamentals=True,
        data_requirement="fundamental:revenue,market_cap",
    )
)
def ps_ratio(ctx: FactorContext) -> pd.DataFrame:
    sp = _ratio(ctx, "sales_to_price")
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / sp.replace(0.0, np.nan)


@register(
    FactorSpec(
        name="value_composite",
        category="value",
        direction=1,
        description=(
            "Equal-weighted z-score of earnings yield, book-to-price, sales-to-price "
            "and FCF yield. Diversifies away single-metric noise."
        ),
        requires_fundamentals=True,
        data_requirement="fundamental:multiple",
    )
)
def value_composite(ctx: FactorContext) -> pd.DataFrame:
    parts = []
    for key in ("earnings_yield", "book_to_price", "sales_to_price", "fcf_yield"):
        try:
            parts.append(_ratio(ctx, key))
        except Exception:  # noqa: BLE001
            continue
    if not parts:
        raise RuntimeError("No value inputs available")
    zs = [
        p.sub(p.mean(axis=1), axis=0).div(p.std(axis=1).replace(0, np.nan), axis=0) for p in parts
    ]
    return pd.concat(zs).groupby(level=0).mean()


__all__ = [
    "earnings_yield",
    "book_to_price",
    "sales_to_price",
    "fcf_yield",
    "ebit_to_ev",
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "value_composite",
]

"""Offline tests for the value factors.

Value factors are the most convention-laden family in the book, and the two
conventions are both easy to invert:

  * **the direction.** ``earnings_yield``, ``book_to_price`` and the rest are
    "higher is cheaper" and carry direction ``+1``; the price ratios are their
    reciprocals and carry direction ``-1``. A factor that is right in value but
    wrong in direction is worse than a missing one, because it is used - the
    optimiser will happily buy the most expensive names in the universe.
  * **the reciprocal.** ``pe_ratio`` must be exactly ``1 / earnings_yield``,
    including at the boundary: a zero yield must become NaN, not ``inf``.

The composite gets its own independent check: its z-scores are recomputed with
plain pandas on a single date and compared, rather than being asserted against
the same concat/groupby expression that produced them.

One regression is pinned here. Both composites used to raise a plain
``RuntimeError`` when no inputs were available, which escaped
``FactorRegistry.compute``'s ``FactorUnavailableError`` handler - so instead of
being reported as unavailable like every other factor, they aborted the call.
The module docstring already promised the unavailable semantics.

Everything is deterministic: no network, fixed seeds, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.factors import value as value_module
from alphaforge.factors.base import REGISTRY, FactorContext, FactorUnavailableError
from alphaforge.factors.value import (
    book_to_price,
    earnings_yield,
    ebit_to_ev,
    fcf_yield,
    pb_ratio,
    pe_ratio,
    ps_ratio,
    sales_to_price,
    value_composite,
)
from alphaforge.features.fundamentals import FundamentalView
from alphaforge.features.panel import MarketPanel, build_panel

DATES = pd.bdate_range("2021-01-04", periods=400)
SYMBOLS = [f"S{i:02d}" for i in range(12)]
VALUE_FACTORS = [
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
# (price-ratio function, the yield it inverts)
RECIPROCALS = [
    (pe_ratio, "earnings_yield"),
    (pb_ratio, "book_to_price"),
    (ps_ratio, "sales_to_price"),
]
RECIPROCAL_IDS = ["pe_ratio", "pb_ratio", "ps_ratio"]


@pytest.fixture(scope="module")
def panel() -> MarketPanel:
    rng = np.random.default_rng(3)
    rows = []
    for sym in SYMBOLS:
        path = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(DATES))))
        for date, price in zip(DATES, path):
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "adj_close": price,
                    "close": price,
                    "volume": 1.0e6,
                    "market_cap": 1.0e9,
                    "industry": "Tech",
                }
            )
    return build_panel(pd.DataFrame(rows))


def _statements(seed: int = 11, drop: tuple[str, ...] = ()) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym in SYMBOLS:
        for report_date in ("2021-06-01", "2022-06-01"):
            rows.append(
                {
                    "symbol": sym,
                    "report_date": pd.Timestamp(report_date),
                    "net_income": float(rng.normal(1.0e8, 1.0e7)),
                    "total_equity": float(rng.normal(1.0e9, 1.0e8)),
                    "revenue": float(rng.normal(5.0e8, 5.0e7)),
                    "operating_cashflow": float(rng.normal(1.2e8, 1.0e7)),
                    "capex": float(rng.normal(3.0e7, 3.0e6)),
                    "gross_profit": float(rng.normal(2.0e8, 2.0e7)),
                    "ebit": float(rng.normal(1.5e8, 1.5e7)),
                    "total_assets": float(rng.normal(2.0e9, 2.0e8)),
                    "total_debt": float(rng.normal(4.0e8, 4.0e7)),
                }
            )
    return pd.DataFrame(rows).drop(columns=list(drop))


def _view(panel: MarketPanel, drop: tuple[str, ...] = ()) -> FundamentalView:
    return FundamentalView.build(
        _statements(drop=drop), panel.dates, panel.symbols, panel.market_cap
    )


@pytest.fixture(scope="module")
def fundamentals(panel: MarketPanel) -> FundamentalView:
    return _view(panel)


@pytest.fixture(scope="module")
def ctx(panel: MarketPanel, fundamentals: FundamentalView) -> FactorContext:
    return FactorContext(panel, fundamentals=fundamentals)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
def test_every_value_factor_is_registered_with_the_right_direction() -> None:
    cheap_is_good = {
        "earnings_yield",
        "book_to_price",
        "sales_to_price",
        "fcf_yield",
        "ebit_to_ev",
        "value_composite",  # a z-score of the yields, so it points the same way
    }
    for name in VALUE_FACTORS:
        spec = REGISTRY.spec(name)
        assert spec.category == "value"
        assert spec.requires_fundamentals is True
        expected = 1 if name in cheap_is_good else -1
        assert spec.direction == expected, f"{name} has the wrong direction"


def test_the_price_ratios_are_reciprocals_of_the_yields() -> None:
    registered = REGISTRY.names()
    for ratio_fn, yield_name in RECIPROCALS:
        assert ratio_fn.__name__ in registered
        assert yield_name in registered


# ----------------------------------------------------------------------
# The raw ratios
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("factor", "derived"),
    [
        (earnings_yield, "earnings_yield"),
        (book_to_price, "book_to_price"),
        (sales_to_price, "sales_to_price"),
        (fcf_yield, "fcf_yield"),
        (ebit_to_ev, "ebit_to_ev"),
    ],
)
def test_each_factor_is_a_thin_wrapper_over_the_point_in_time_ratio(
    ctx: FactorContext, factor, derived: str
) -> None:
    got = factor(ctx)
    expected = ctx.fundamentals.ratio(derived)
    pd.testing.assert_frame_equal(got, expected)


@pytest.mark.parametrize(("ratio", "derived"), RECIPROCALS, ids=RECIPROCAL_IDS)
def test_a_price_ratio_is_exactly_the_reciprocal(ctx: FactorContext, ratio, derived: str) -> None:
    got = ratio(ctx)
    underlying = ctx.fundamentals.ratio(derived)
    finite = underlying.notna() & (underlying != 0)
    np.testing.assert_allclose(
        got[finite].to_numpy(), (1.0 / underlying[finite]).to_numpy(), rtol=1e-12
    )


@pytest.mark.parametrize(("ratio", "_derived"), RECIPROCALS, ids=RECIPROCAL_IDS)
def test_a_zero_yield_becomes_nan_not_infinity(ctx: FactorContext, ratio, _derived) -> None:
    """``1 / 0`` is ``inf``, and an ``inf`` factor poisons every ranking."""
    got = ratio(ctx)
    assert np.isfinite(got.to_numpy(dtype=float)[np.isfinite(got.to_numpy(dtype=float))]).all()
    assert not np.isinf(got.to_numpy(dtype=float)).any()


# ----------------------------------------------------------------------
# Missing data
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", VALUE_FACTORS)
def test_a_missing_fundamental_table_raises_the_unavailable_error(
    panel: MarketPanel, name: str
) -> None:
    """Regression: the composites raised a plain ``RuntimeError`` here.

    ``FactorUnavailableError`` is what ``FactorRegistry.compute`` catches, so a
    plain ``RuntimeError`` escaped the handler and aborted the call instead of
    reporting the factor as unavailable like every other one.
    """
    bare = FactorContext(panel, fundamentals=None)
    assert REGISTRY.spec(name).requires_fundamentals
    with pytest.raises(FactorUnavailableError):
        getattr(value_module, name)(bare)


@pytest.mark.parametrize("name", VALUE_FACTORS)
def test_the_registry_reports_the_composites_as_unavailable_not_as_a_crash(
    panel: MarketPanel, name: str
) -> None:
    bare = FactorContext(panel, fundamentals=None)
    factor = REGISTRY.compute(name, bare)  # must not raise
    assert factor.coverage() == 0.0


def test_a_value_composite_ignores_an_input_it_cannot_build(
    panel: MarketPanel, fundamentals: FundamentalView
) -> None:
    """Dropping one statement line must cost one component, not the whole factor."""
    full = value_composite(FactorContext(panel, fundamentals=fundamentals))
    partial_view = _view(panel, drop=("revenue",))
    partial = value_composite(FactorContext(panel, fundamentals=partial_view))
    assert partial.notna().any().any()
    assert not partial.equals(full)


# ----------------------------------------------------------------------
# The composite
# ----------------------------------------------------------------------
def test_the_composite_is_the_mean_of_cross_sectional_z_scores(
    panel: MarketPanel, fundamentals: FundamentalView, ctx: FactorContext
) -> None:
    """Recomputed with plain pandas on one date, not with the same expression."""
    got = value_composite(ctx)
    date = DATES[-1]
    keys = ("earnings_yield", "book_to_price", "sales_to_price", "fcf_yield")
    raw = pd.DataFrame({k: fundamentals.ratio(k).loc[date] for k in keys})
    z = raw.sub(raw.mean(axis=0), axis=1).div(raw.std(axis=0, ddof=1), axis=1)
    expected = z.mean(axis=1)
    np.testing.assert_allclose(
        got.loc[date].reindex(expected.index).to_numpy(),
        expected.to_numpy(),
        rtol=1e-10,
        equal_nan=True,
    )


def test_the_composite_is_centred_on_each_date(ctx: FactorContext) -> None:
    got = value_composite(ctx)
    row = got.iloc[-1].dropna()
    assert row.mean() == pytest.approx(0.0, abs=1e-12)


def test_the_composite_keeps_a_name_that_only_some_inputs_cover(
    panel: MarketPanel, fundamentals: FundamentalView
) -> None:
    """Averaging the available z-scores is the point of a composite."""
    ctx = FactorContext(panel, fundamentals=fundamentals)
    date = DATES[-1]
    present = fundamentals.ratio("earnings_yield").loc[date].notna()
    got = value_composite(ctx).loc[date]
    assert got[present].notna().any()


def test_the_composite_is_unavailable_when_no_inputs_exist(panel: MarketPanel) -> None:
    bare = FactorContext(panel, fundamentals=None)
    with pytest.raises(FactorUnavailableError, match="No value inputs available"):
        value_composite(bare)

"""Offline tests for the quality / profitability factors.

Quality is the factor group most exposed to look-ahead, so the point-in-time
behaviour is asserted directly: a ratio on a date must equal the ratio built
from the statement whose ``report_date`` is the latest one on or before it, and
nothing later.

Two conventions are pinned here because both are easy to invert:

  * ``accruals`` and ``low_leverage`` carry direction ``-1``. ``low_leverage`` in
    particular returns the raw debt/assets ratio - it is *not* sign-flipped - so
    a reader who trusts the old description ("signed so that higher = safer")
    would flip the direction and get the sign backwards.
  * ``earnings_quality`` is ``(ocf - ni) / assets`` and ``accruals`` is its
    negation, so the two must be exact mirror images. If one of them ever drifts,
    the pair stops being a consistency check on the accrual line.

One regression is pinned. ``quality_composite`` used to raise a plain
``RuntimeError`` when no inputs were available, which escaped
``FactorRegistry.compute``'s ``FactorUnavailableError`` handler - so instead of
being reported as unavailable like every other factor, it aborted the call.

Everything is deterministic: no network, fixed seeds, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.factors import quality as quality_module
from alphaforge.factors.base import REGISTRY, FactorContext, FactorUnavailableError
from alphaforge.factors.quality import (
    accruals,
    earnings_quality,
    low_leverage,
    quality_composite,
    roa,
    roe,
)
from alphaforge.features.fundamentals import FundamentalView
from alphaforge.features.panel import MarketPanel, build_panel

DATES = pd.bdate_range("2021-01-04", periods=400)
SYMBOLS = [f"S{i:02d}" for i in range(12)]
QUALITY_FACTORS = [
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
# Raw value points the same way as the expected return...
POSITIVE_DIRECTION = {
    "roe",
    "roa",
    "gross_profitability",
    "asset_turnover",
    "earnings_quality",
    "gross_margin",
    "quality_composite",
}
# ...and these are bearish at the top of the cross-section.
NEGATIVE_DIRECTION = {"accruals", "low_leverage"}

DERIVED = {
    "roe": "roe",
    "roa": "roa",
    "gross_profitability": "gross_profitability",
    "asset_turnover": "asset_turnover",
    "earnings_quality": "earnings_quality",
    "accruals": "accruals",
    "gross_margin": "gross_margin",
    "low_leverage": "leverage",
}


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


def _statements(seed: int = 23) -> pd.DataFrame:
    """Two annual statements per name, so the point-in-time join has work to do."""
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
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fundamentals(panel: MarketPanel) -> FundamentalView:
    return FundamentalView.build(_statements(), panel.dates, panel.symbols, panel.market_cap)


@pytest.fixture(scope="module")
def ctx(panel: MarketPanel, fundamentals: FundamentalView) -> FactorContext:
    return FactorContext(panel, fundamentals=fundamentals)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
def test_every_quality_factor_is_registered_with_the_right_direction() -> None:
    assert set(QUALITY_FACTORS) == POSITIVE_DIRECTION | NEGATIVE_DIRECTION
    for name in QUALITY_FACTORS:
        spec = REGISTRY.spec(name)
        assert spec.category == "quality"
        assert spec.requires_fundamentals is True
        expected = 1 if name in POSITIVE_DIRECTION else -1
        assert spec.direction == expected, f"{name} has the wrong direction"


def test_low_leverage_describes_itself_as_unsiged_leverage() -> None:
    """The old text said "higher = safer" while the value was raw leverage."""
    description = REGISTRY.spec("low_leverage").description
    assert "not" in description and "sign" in description


# ----------------------------------------------------------------------
# The raw ratios
# ----------------------------------------------------------------------
@pytest.mark.parametrize(("name", "derived"), sorted(DERIVED.items()))
def test_each_factor_is_a_thin_wrapper_over_the_point_in_time_ratio(
    ctx: FactorContext, name: str, derived: str
) -> None:
    got = getattr(quality_module, name)(ctx)
    pd.testing.assert_frame_equal(got, ctx.fundamentals.ratio(derived))


def test_accruals_is_the_mirror_image_of_earnings_quality(ctx: FactorContext) -> None:
    """``eq = (ocf - ni)/a`` and ``accruals = (ni - ocf)/a`` - exact opposites."""
    eq = earnings_quality(ctx)
    ac = accruals(ctx)
    both = eq.notna() & ac.notna()
    np.testing.assert_allclose(eq[both].to_numpy(), -ac[both].to_numpy(), rtol=1e-12, atol=1e-18)


def test_low_leverage_returns_leverage_rather_than_its_negative(ctx: FactorContext) -> None:
    """The direction carries the sign; a negated value here would double-flip it."""
    got = low_leverage(ctx)
    expected = ctx.fundamentals.ratio("leverage")
    pd.testing.assert_frame_equal(got, expected)
    assert (got.dropna(how="all").to_numpy() >= -1e-12).all()


# ----------------------------------------------------------------------
# Point in time
# ----------------------------------------------------------------------
def test_a_ratio_only_uses_a_statement_that_was_already_public(ctx: FactorContext) -> None:
    """Before the first report date the factor is empty; after it, it is populated."""
    got = roe(ctx)
    first_report = pd.Timestamp("2021-06-01")
    assert got.loc[got.index < first_report].isna().all().all()
    assert got.loc[got.index >= first_report].notna().any().any()


def test_the_ratio_changes_when_the_second_statement_lands(ctx: FactorContext) -> None:
    """A stale statement must not persist once a newer one is released."""
    got = roe(ctx)
    second_report = pd.Timestamp("2022-06-01")
    before = got.loc[got.index < second_report].iloc[-1]
    after = got.loc[got.index >= second_report].iloc[0]
    assert not np.allclose(before.to_numpy(), after.to_numpy(), equal_nan=True)


def test_a_ratio_is_nan_rather_than_zero_when_the_denominator_vanishes(
    panel: MarketPanel,
) -> None:
    statements = _statements()
    statements.loc[statements["report_date"] == pd.Timestamp("2022-06-01"), "total_assets"] = 0.0
    view = FundamentalView.build(statements, panel.dates, panel.symbols, panel.market_cap)
    got = roa(FactorContext(panel, fundamentals=view))
    tail = got.loc[got.index >= pd.Timestamp("2022-06-01")]
    assert tail.isna().all().all(), "a zero denominator must be NaN, not inf"


# ----------------------------------------------------------------------
# Missing data
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", QUALITY_FACTORS)
def test_a_missing_fundamental_table_raises_the_unavailable_error(
    panel: MarketPanel, name: str
) -> None:
    """Regression: the composite raised a plain ``RuntimeError`` here."""
    bare = FactorContext(panel, fundamentals=None)
    with pytest.raises(FactorUnavailableError):
        getattr(quality_module, name)(bare)


@pytest.mark.parametrize("name", QUALITY_FACTORS)
def test_the_registry_reports_every_quality_factor_as_unavailable(
    panel: MarketPanel, name: str
) -> None:
    factor = REGISTRY.compute(name, FactorContext(panel, fundamentals=None))
    assert factor.coverage() == 0.0


# ----------------------------------------------------------------------
# The composite
# ----------------------------------------------------------------------
def test_the_composite_is_the_mean_of_cross_sectional_z_scores(
    fundamentals: FundamentalView, ctx: FactorContext
) -> None:
    """Recomputed with plain pandas on one date, not with the same expression."""
    got = quality_composite(ctx)
    date = DATES[-1]
    keys = ("roe", "roa", "gross_profitability", "earnings_quality")
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
    row = quality_composite(ctx).iloc[-1].dropna()
    assert row.mean() == pytest.approx(0.0, abs=1e-12)


def test_the_composite_is_unavailable_when_no_inputs_exist(panel: MarketPanel) -> None:
    with pytest.raises(FactorUnavailableError, match="No quality inputs available"):
        quality_composite(FactorContext(panel, fundamentals=None))


def test_the_composite_survives_a_statement_line_that_cannot_be_built(
    panel: MarketPanel,
) -> None:
    """Losing one input must cost one component, not the whole factor."""
    statements = _statements().drop(columns=["gross_profit"])
    view = FundamentalView.build(statements, panel.dates, panel.symbols, panel.market_cap)
    got = quality_composite(FactorContext(panel, fundamentals=view))
    assert got.notna().any().any()

"""Offline tests for the liquidity and size factors.

``direction`` is not decoration in this codebase: ``FactorPreprocessor.process``
does ``if factor.spec.direction == -1: df = -df`` before the panel reaches the
model. A factor whose direction is wrong is therefore not merely mislabelled -
it is fed to the model **inverted**, so the model is trained to buy exactly what
the factor says to sell.

That makes the internal consistency of the directions a testable property, and
the module currently fails it. Six of these factors measure one construct,
liquidity, from both sides:

  * more liquid = higher raw value: ``adv_21d``, ``log_adv_21d``,
    ``dollar_volume_ratio``, ``turnover_21d``
  * less liquid = higher raw value: ``amihud_illiquidity``, ``zero_trading_days``

The module docstring says these are alphas for the *illiquidity premium*, and
``log_market_cap`` / ``log_price`` / ``amihud_illiquidity`` / ``turnover_21d``
all follow it. The other four do not, and they contradict each other:
``turnover_21d`` is -1 while ``adv_21d`` is +1 although both rise with liquidity,
and ``amihud_illiquidity`` is +1 while ``zero_trading_days`` is -1 although both
rise with illiquidity.

``test_the_liquidity_directions_agree_with_each_other`` records that as a
**strict xfail**: the expectation is written down, the failure is documented,
and fixing the directions turns it into an XPASS which fails the suite until the
marker is removed. No direction is changed here - which of the two readings the
author intended for each factor is a strategy decision, not a bug fix.

Everything is deterministic: no network, fixed seeds, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.factors.base import REGISTRY, FactorContext
from alphaforge.factors.liquidity import (
    adv_21d,
    amihud_illiquidity,
    dollar_volume_ratio,
    log_adv_21d,
    log_market_cap,
    log_price,
    turnover_21d,
    zero_trading_days,
)
from alphaforge.features.panel import MarketPanel, build_panel

DATES = pd.bdate_range("2022-01-03", periods=320)
SYMBOLS = [f"S{i:02d}" for i in range(10)]
LIQUIDITY_FACTORS = [
    "log_market_cap",
    "log_price",
    "adv_21d",
    "log_adv_21d",
    "turnover_21d",
    "amihud_illiquidity",
    "dollar_volume_ratio",
    "zero_trading_days",
]
# Raw value rises with liquidity...
RISES_WITH_LIQUIDITY = {"adv_21d", "log_adv_21d", "dollar_volume_ratio", "turnover_21d"}
# ...and these rise with *il*liquidity.
RISES_WITH_ILLIQUIDITY = {"amihud_illiquidity", "zero_trading_days"}


def _panel(volume: np.ndarray | None = None, seed: int = 5) -> MarketPanel:
    rng = np.random.default_rng(seed)
    if volume is None:
        volume = rng.uniform(1.0e5, 5.0e6, size=(len(DATES), len(SYMBOLS)))
    rows = []
    for j, sym in enumerate(SYMBOLS):
        path = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(DATES))))
        for i, (date, price) in enumerate(zip(DATES, path)):
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "adj_close": price,
                    "close": price,
                    "volume": float(volume[i, j]),
                    "market_cap": 1.0e9 * (1.0 + 0.1 * j),
                    "industry": "Tech",
                }
            )
    return build_panel(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def panel() -> MarketPanel:
    return _panel()


@pytest.fixture(scope="module")
def ctx(panel: MarketPanel) -> FactorContext:
    return FactorContext(panel)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
def test_every_factor_is_registered_in_a_size_or_liquidity_bucket() -> None:
    for name in LIQUIDITY_FACTORS:
        spec = REGISTRY.spec(name)
        assert spec.category in {"size", "liquidity"}
        assert spec.direction in (1, -1)


def test_the_size_factors_follow_the_small_cap_premium() -> None:
    """Bigger, pricier names must rank as *less* attractive."""
    assert REGISTRY.spec("log_market_cap").direction == -1
    assert REGISTRY.spec("log_price").direction == -1


# ----------------------------------------------------------------------
# The raw values
# ----------------------------------------------------------------------
def test_log_market_cap_is_the_log_of_capitalisation(ctx: FactorContext) -> None:
    got = log_market_cap(ctx)
    np.testing.assert_allclose(got.to_numpy(), np.log(ctx.market_cap.to_numpy()), rtol=1e-12)


def test_log_market_cap_turns_a_zero_into_nan() -> None:
    """A fresh panel: the module-scoped one must stay pristine for the others."""
    ctx = FactorContext(_panel())
    ctx.market_cap.iloc[10, 0] = 0.0
    got = log_market_cap(ctx)
    assert np.isnan(got.iloc[10, 0])
    assert got.iloc[10, 1:].notna().all()


def test_log_price_is_the_log_of_the_adjusted_close(ctx: FactorContext) -> None:
    np.testing.assert_allclose(log_price(ctx).to_numpy(), np.log(ctx.close.to_numpy()), rtol=1e-12)


def test_adv_is_a_trailing_dollar_volume_mean(panel: MarketPanel, ctx: FactorContext) -> None:
    """An explicit numpy slice over rows 100..120, not the same pandas call."""
    got = adv_21d(ctx)
    window = panel.dollar_volume.iloc[100:121, 0].to_numpy()
    assert got.iloc[120, 0] == pytest.approx(float(window.mean()), rel=1e-12)
    assert got.iloc[:9].isna().all().all(), "min_periods=10 means row 9 is the first valid one"
    assert got.iloc[9:].notna().all().all()


def test_log_adv_is_the_log_of_adv(ctx: FactorContext) -> None:
    adv = adv_21d(ctx)
    got = log_adv_21d(ctx)
    both = adv.notna() & (adv > 0)
    np.testing.assert_allclose(got[both].to_numpy(), np.log(adv[both]).to_numpy(), rtol=1e-12)


def test_turnover_divides_volume_by_implied_shares(panel: MarketPanel, ctx: FactorContext) -> None:
    got = turnover_21d(ctx)
    shares = panel.market_cap / panel.close
    expected = panel.volume.rolling(21, min_periods=10).mean() / shares
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-10, equal_nan=True)


def test_amihud_is_absolute_return_per_dollar_traded(ctx: FactorContext) -> None:
    got = amihud_illiquidity(ctx)
    expected = (ctx.returns.abs() / ctx.panel.dollar_volume).rolling(
        21, min_periods=10
    ).mean() * 1e6
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-10, equal_nan=True)


def test_dollar_volume_ratio_is_short_adv_over_long_adv(
    panel: MarketPanel, ctx: FactorContext
) -> None:
    got = dollar_volume_ratio(ctx)
    dv = panel.dollar_volume
    expected = dv.rolling(21, min_periods=10).mean() / dv.rolling(252, min_periods=126).mean()
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-10, equal_nan=True)
    assert got.iloc[:125].isna().all().all(), "min_periods=126 means row 125 is the first valid one"
    assert got.iloc[125:].notna().all().all()


# ----------------------------------------------------------------------
# Zero trading days
# ----------------------------------------------------------------------
def test_zero_trading_days_counts_the_dead_sessions() -> None:
    volume = np.full((len(DATES), len(SYMBOLS)), 1.0e6)
    volume[:, 0] = 0.0  # S00 never trades
    volume[-21:, 1] = 0.0  # S01 went quiet for the last month
    got = zero_trading_days(FactorContext(_panel(volume)))
    assert got["S00"].iloc[-1] == pytest.approx(1.0)
    assert got["S01"].iloc[-1] == pytest.approx(1.0)
    assert got["S02"].iloc[-1] == pytest.approx(0.0)
    assert got["S01"].iloc[-30] == pytest.approx(0.0)


def test_zero_trading_days_is_a_fraction(ctx: FactorContext) -> None:
    got = zero_trading_days(ctx)
    values = got.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    assert ((finite >= 0.0) & (finite <= 1.0)).all()


def test_a_missing_volume_is_treated_as_a_dead_session() -> None:
    """Documented choice: ``fillna(0)`` counts no-data as no-trading.

    Only the tail is blanked - a column that is NaN throughout is dropped when
    the panel is built, so it would never reach the factor at all.
    """
    volume = np.full((len(DATES), len(SYMBOLS)), 1.0e6)
    volume[-21:, 0] = np.nan
    got = zero_trading_days(FactorContext(_panel(volume)))
    assert got["S00"].iloc[-1] == pytest.approx(1.0)
    assert got["S00"].iloc[-30] == pytest.approx(0.0)
    assert got["S01"].iloc[-1] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# No look-ahead
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "factor",
    [
        adv_21d,
        log_adv_21d,
        turnover_21d,
        amihud_illiquidity,
        dollar_volume_ratio,
        zero_trading_days,
    ],
)
def test_liquidity_measures_never_see_the_future(factor) -> None:
    baseline = _panel()
    tampered = _panel()
    tampered.dollar_volume.iloc[300:] = 1.0
    tampered.volume.iloc[300:] = 1.0

    after = factor(FactorContext(tampered)).iloc[:300].to_numpy(dtype=float)
    before = factor(FactorContext(baseline)).iloc[:300].to_numpy(dtype=float)
    np.testing.assert_allclose(after, before, equal_nan=True)


# ----------------------------------------------------------------------
# The direction inconsistency
# ----------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason=(
        "adv_21d/log_adv_21d/dollar_volume_ratio are +1 while turnover_21d is -1 "
        "although all four rise with liquidity, and zero_trading_days is -1 while "
        "amihud_illiquidity is +1 although both rise with illiquidity. "
        "FactorPreprocessor flips the sign on direction == -1, so the model is "
        "being fed four inverted liquidity signals. Remove this marker once the "
        "directions agree."
    ),
)
def test_the_liquidity_directions_agree_with_each_other() -> None:
    liquid = {REGISTRY.spec(n).direction for n in RISES_WITH_LIQUIDITY}
    illiquid = {REGISTRY.spec(n).direction for n in RISES_WITH_ILLIQUIDITY}
    assert len(liquid) == 1, f"measures of liquidity disagree: {sorted(liquid)}"
    assert len(illiquid) == 1, f"measures of illiquidity disagree: {sorted(illiquid)}"
    assert liquid != illiquid, "liquidity and illiquidity cannot point the same way"


def test_the_illiquidity_premium_is_the_named_framing() -> None:
    """The docstring, not this test, is the authority on the intended sign."""
    import alphaforge.factors.liquidity as liquidity_module

    assert "illiquidity premium" in liquidity_module.__doc__.lower()

"""Offline tests for the momentum / reversal factors.

Momentum is the most bug-prone factor family in the book, because the classic
specification carries two easily-inverted conventions:

  * **the skip.** ``mom_12_1`` is the return from ``t - 271`` to ``t - 21``, not
    ``t - 250`` to ``t``. Getting it wrong leaves the most recent month - the
    part contaminated by short-term reversal and the bid-ask bounce - inside the
    signal, which is exactly what the specification exists to exclude.
  * **the sign.** Reversal factors carry direction ``-1``: the raw value is still
    the trailing return, and the *direction* is what makes a high past return a
    sell.

So the tests are built to discriminate rather than to agree. A price series with
a crash in the last 21 days must leave ``mom_12_1`` untouched while moving
``mom_20d``. And the residual-momentum check is the strongest available: a stock
whose returns *are* the market's has beta 1 and zero alpha, so the factor must
come out at zero - anything else means the beta or the intercept is wrong.

Everything is deterministic: no network, fixed seeds, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.factors.base import REGISTRY, FactorContext
from alphaforge.factors.momentum import (
    _momentum,
    industry_momentum,
    mom_6_1,
    mom_12_1,
    mom_20d,
    mom_60d,
    mom_120d,
    residual_momentum,
    rev_5d,
    rev_21d,
)
from alphaforge.features.panel import MarketPanel, build_panel

DATES = pd.bdate_range("2021-01-04", periods=400)
SYMBOLS = ["AAA", "BBB", "CCC"]
MOMENTUM_FACTORS = ["mom_20d", "mom_60d", "mom_120d", "mom_12_1", "mom_6_1", "residual_momentum"]
REVERSAL_FACTORS = ["rev_5d", "rev_21d"]


def _panel(prices: dict[str, np.ndarray], industry: dict[str, str] | None = None) -> MarketPanel:
    rows = []
    for sym, path in prices.items():
        label = (industry or {}).get(sym, "Tech")
        for date, price in zip(DATES[: len(path)], path):
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "adj_close": price,
                    "close": price,
                    "volume": 1.0e6,
                    "market_cap": 1.0e9,
                    "industry": label,
                }
            )
    return build_panel(pd.DataFrame(rows))


def _ramp(rate: float, n: int = len(DATES), base: float = 100.0) -> np.ndarray:
    return base * np.exp(rate * np.arange(n))


def _flat_panel(rate: float = 0.001) -> MarketPanel:
    return _panel({sym: _ramp(rate) for sym in SYMBOLS})


def _random_returns(seed: int = 13) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, size=len(DATES)), index=DATES)


def _panel_from_returns(rets: pd.DataFrame) -> MarketPanel:
    prices = 100.0 * (1.0 + rets.fillna(0.0)).cumprod()
    return _panel({sym: prices[sym].to_numpy() for sym in rets.columns})


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
def test_momentum_factors_point_the_same_way() -> None:
    for name in MOMENTUM_FACTORS:
        spec = REGISTRY.spec(name)
        assert spec.category == "momentum"
        assert spec.direction == 1, f"{name} should reward past strength"


def test_reversal_factors_are_registered_with_the_opposite_sign() -> None:
    for name in REVERSAL_FACTORS:
        spec = REGISTRY.spec(name)
        assert spec.category == "reversal"
        assert spec.direction == -1, f"{name} should punish past strength"


# ----------------------------------------------------------------------
# The raw trailing return
# ----------------------------------------------------------------------
def test_momentum_is_the_trailing_total_return() -> None:
    panel = _flat_panel()
    got = mom_20d(FactorContext(panel))["AAA"]
    close = panel.close["AAA"]
    for t in (100, 250, 399):
        assert got.iloc[t] == pytest.approx(close.iloc[t] / close.iloc[t - 20] - 1.0, rel=1e-12)


@pytest.mark.parametrize(
    ("factor", "lookback"),
    [(mom_20d, 20), (mom_60d, 60), (mom_120d, 120), (rev_5d, 5), (rev_21d, 21)],
)
def test_each_lookback_uses_its_own_window(factor, lookback: int) -> None:
    panel = _flat_panel()
    got = factor(FactorContext(panel))["AAA"]
    close = panel.close["AAA"]
    assert got.iloc[-1] == pytest.approx(
        close.iloc[-1] / close.iloc[-1 - lookback] - 1.0, rel=1e-12
    )
    assert got.iloc[:lookback].isna().all()


def test_rising_prices_give_positive_momentum_and_falling_negative() -> None:
    assert mom_20d(FactorContext(_flat_panel(0.001)))["AAA"].iloc[-1] > 0
    assert mom_20d(FactorContext(_flat_panel(-0.001)))["AAA"].iloc[-1] < 0


def test_the_reversal_value_is_still_the_trailing_return() -> None:
    """Direction carries the sign; the raw number must not be pre-negated."""
    panel = _flat_panel(0.002)
    assert rev_5d(FactorContext(panel))["AAA"].iloc[-1] > 0
    assert rev_21d(FactorContext(panel))["AAA"].iloc[-1] > 0


# ----------------------------------------------------------------------
# The skip
# ----------------------------------------------------------------------
def test_the_skip_helper_measures_between_the_two_shifted_prices() -> None:
    close = pd.DataFrame({"AAA": _ramp(0.001)}, index=DATES)
    got = _momentum(close, lookback=250, skip=21)
    expected = close["AAA"].shift(21) / close["AAA"].shift(271) - 1.0
    np.testing.assert_allclose(got["AAA"].to_numpy(), expected.to_numpy(), equal_nan=True)


def test_mom_12_1_spans_two_hundred_and_fifty_days_ending_a_month_ago() -> None:
    panel = _flat_panel()
    close = panel.close["AAA"]
    got = mom_12_1(FactorContext(panel))["AAA"]
    t = len(DATES) - 1
    assert got.iloc[t] == pytest.approx(close.iloc[t - 21] / close.iloc[t - 271] - 1.0, rel=1e-12)
    # On a constant-rate ramp the skipped and unskipped windows coincide, so the
    # discrimination is made by ``test_the_skip_actually_excludes_...`` instead.
    assert got.iloc[:271].isna().all()


def test_mom_6_1_skips_a_month_of_a_half_year_window() -> None:
    panel = _flat_panel()
    close = panel.close["AAA"]
    got = mom_6_1(FactorContext(panel))["AAA"]
    t = len(DATES) - 1
    assert got.iloc[t] == pytest.approx(close.iloc[t - 21] / close.iloc[t - 147] - 1.0, rel=1e-12)


def test_the_skip_actually_excludes_the_most_recent_month() -> None:
    """A crash confined to the last 21 days must not reach ``mom_12_1``."""
    calm = _flat_panel(0.001)
    crashed = _panel({sym: _ramp(0.001) for sym in SYMBOLS})
    # Halve the last 20 days only: scaling a whole 20-day block would leave the
    # ratio across it unchanged and prove nothing.
    for sym in SYMBOLS:
        crashed.close.loc[DATES[-20] :, sym] = crashed.close.loc[DATES[-20] :, sym] * 0.5

    skipped = mom_12_1(FactorContext(crashed))["AAA"]
    baseline = mom_12_1(FactorContext(calm))["AAA"]
    np.testing.assert_allclose(
        skipped.iloc[:-20].to_numpy(), baseline.iloc[:-20].to_numpy(), equal_nan=True
    )

    # ...while the unskipped window does see it.
    assert mom_20d(FactorContext(crashed))["AAA"].iloc[-1] < -0.4
    assert mom_12_1(FactorContext(crashed))["AAA"].iloc[-1] == pytest.approx(
        baseline.iloc[-1], rel=1e-12
    )


# ----------------------------------------------------------------------
# No look-ahead
# ----------------------------------------------------------------------
@pytest.mark.parametrize("factor", [mom_20d, mom_60d, mom_120d, mom_12_1, mom_6_1, rev_5d, rev_21d])
def test_momentum_never_sees_the_future(factor) -> None:
    baseline_panel = _flat_panel()
    tampered = _panel({sym: _ramp(0.001) for sym in SYMBOLS})
    tampered.close.iloc[380:] = 1.0

    after = factor(FactorContext(tampered))["AAA"].iloc[:380].to_numpy()
    before = factor(FactorContext(baseline_panel))["AAA"].iloc[:380].to_numpy()
    np.testing.assert_allclose(after, before, equal_nan=True)


# ----------------------------------------------------------------------
# Residual momentum
# ----------------------------------------------------------------------
def test_residual_momentum_is_zero_when_every_stock_is_the_market() -> None:
    """Beta 1 and zero alpha: anything non-zero means the regression is wrong."""
    market = _random_returns()
    panel = _panel_from_returns(pd.DataFrame({sym: market for sym in SYMBOLS}))
    got = residual_momentum(FactorContext(panel))
    assert got.abs().max().max() < 1e-12


def test_residual_momentum_ignores_the_market_component() -> None:
    """Adding a common market move must not change a stock's residual alpha."""
    rng = np.random.default_rng(17)
    market = pd.Series(rng.normal(0.0004, 0.01, size=len(DATES)), index=DATES)
    idiosyncratic = pd.Series(rng.normal(0.0, 0.004, size=len(DATES)), index=DATES)
    quiet = _panel_from_returns(pd.DataFrame({sym: market + idiosyncratic for sym in SYMBOLS}))
    loud = _panel_from_returns(pd.DataFrame({sym: 2.0 * market + idiosyncratic for sym in SYMBOLS}))
    quiet_val = residual_momentum(FactorContext(quiet))["AAA"].dropna()
    loud_val = residual_momentum(FactorContext(loud))["AAA"].dropna()
    common = quiet_val.index.intersection(loud_val.index)
    np.testing.assert_allclose(quiet_val[common].to_numpy(), loud_val[common].to_numpy(), atol=1e-9)


def test_residual_momentum_is_positive_for_a_stock_with_its_own_trend() -> None:
    rng = np.random.default_rng(19)
    market = pd.Series(rng.normal(0.0, 0.01, size=len(DATES)), index=DATES)
    rets = pd.DataFrame({sym: market for sym in SYMBOLS})
    rets["AAA"] = market + 0.002  # a persistent idiosyncratic drift
    got = residual_momentum(FactorContext(_panel_from_returns(rets)))["AAA"].iloc[-1]
    assert got > 0


def test_residual_momentum_skips_the_most_recent_month() -> None:
    rng = np.random.default_rng(23)
    rets = pd.DataFrame(
        {sym: pd.Series(rng.normal(0.0004, 0.01, size=len(DATES)), index=DATES) for sym in SYMBOLS}
    )
    baseline_panel = _panel_from_returns(rets)

    tampered = rets.copy()
    tampered.iloc[-20:] = 0.5  # only the skipped month moves
    after = residual_momentum(FactorContext(_panel_from_returns(tampered)))["AAA"]
    before = residual_momentum(FactorContext(baseline_panel))["AAA"]
    assert after.iloc[-1] == pytest.approx(before.iloc[-1], rel=1e-12)


# ----------------------------------------------------------------------
# Industry momentum
# ----------------------------------------------------------------------
def test_a_single_industry_reduces_to_the_equal_weighted_portfolio() -> None:
    market = _random_returns()
    panel = _panel_from_returns(pd.DataFrame({sym: market for sym in SYMBOLS}))
    got = industry_momentum(FactorContext(panel))
    expected = (
        (1.0 + panel.returns.mean(axis=1).fillna(0.0)).rolling(60).apply(np.prod, raw=True) - 1.0
    ).shift(1)
    np.testing.assert_allclose(got["AAA"].to_numpy(), expected.to_numpy(), equal_nan=True)


def test_members_of_an_industry_share_one_value() -> None:
    panel = _flat_panel()
    got = industry_momentum(FactorContext(panel))
    assert got.iloc[-1].nunique() == 1


def test_a_strong_industry_outranks_a_weak_one() -> None:
    panel = _panel(
        {"AAA": _ramp(0.002), "BBB": _ramp(0.002), "CCC": _ramp(-0.002)},
        industry={"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"},
    )
    got = industry_momentum(FactorContext(panel))
    assert got["AAA"].iloc[-1] > 0 > got["CCC"].iloc[-1]


def test_industry_momentum_lags_by_a_day() -> None:
    """Today's return is not part of today's signal."""
    market = _random_returns()
    rets = pd.DataFrame({sym: market for sym in SYMBOLS})
    baseline_panel = _panel_from_returns(rets)
    tampered = rets.copy()
    tampered.iloc[-1] = 0.9  # only the final day moves

    after = industry_momentum(FactorContext(_panel_from_returns(tampered)))["AAA"]
    before = industry_momentum(FactorContext(baseline_panel))["AAA"]
    assert after.iloc[-1] == pytest.approx(before.iloc[-1], rel=1e-12)


def test_industry_momentum_needs_a_full_window() -> None:
    short = _panel({sym: _ramp(0.001, n=40) for sym in SYMBOLS})
    got = industry_momentum(FactorContext(short))
    assert got.isna().all().all()

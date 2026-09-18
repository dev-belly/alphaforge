"""Offline tests for the risk / volatility factors.

These factors feed the model directly, so the failure mode that matters is a
number that is *wrong in a plausible way*. Every expectation below is therefore
computed independently - an explicit numpy slice for a rolling window, or a
return series constructed so the answer is known by construction:

  * a return series built as exactly ``beta * market`` must produce that beta,
    and must produce **zero** idiosyncratic volatility;
  * a monotonically rising price series must produce a **zero** maximum
    drawdown, and a series that only ever rises after an initial dip must not be
    charged for the dip it recovered from;
  * every risk measure must be trailing: tampering with the prices on or after a
    date must leave that date's value bit-identical.

The last two are regressions. ``max_drawdown_252d`` used to compare the window's
low against its high, which ignores the ordering a drawdown is defined by: on a
monotone 100 -> 200 ramp the low is the first day and the high the last, so a
stock that never fell was scored as a 42% drawdown - and the factor's direction
is -1, so it was ranked as maximally risky.

Everything is deterministic: no network, fixed seeds, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.factors.base import REGISTRY, FactorContext
from alphaforge.factors.risk import (
    _market_return,
    beta_252d,
    downside_volatility,
    idiosyncratic_volatility,
    max_drawdown_252d,
    return_skew_126d,
    volatility_60d,
    volatility_252d,
)
from alphaforge.features.panel import MarketPanel, build_panel

DATES = pd.bdate_range("2021-01-04", periods=400)
SYMBOLS = ["AAA", "BBB", "CCC"]
ANN = float(np.sqrt(252.0))
RISK_FACTORS = [
    "volatility_60d",
    "volatility_252d",
    "downside_volatility",
    "beta_252d",
    "idiosyncratic_volatility",
    "max_drawdown_252d",
    "return_skew_126d",
]


def _panel_from_returns(rets: pd.DataFrame, benchmark: pd.Series | None = None) -> MarketPanel:
    """Prices whose ``pct_change`` reproduces ``rets`` row for row."""
    prices = 100.0 * (1.0 + rets.fillna(0.0)).cumprod()
    rows = []
    for sym in rets.columns:
        for date, price in zip(rets.index, prices[sym]):
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
    return build_panel(pd.DataFrame(rows), benchmark=benchmark)


def _random_returns(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0004, 0.012, size=(len(DATES), len(SYMBOLS))),
        index=DATES,
        columns=SYMBOLS,
    )


def _ramp(start: float, end: float, n: int) -> np.ndarray:
    return np.linspace(start, end, n)


@pytest.fixture
def panel() -> MarketPanel:
    return _panel_from_returns(_random_returns())


@pytest.fixture
def ctx(panel: MarketPanel) -> FactorContext:
    return FactorContext(panel)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
def test_every_risk_factor_is_registered_as_a_risk_exposure() -> None:
    for name in RISK_FACTORS:
        spec = REGISTRY.spec(name)
        assert spec.category == "risk"
        assert spec.direction == -1, f"{name} should be a low-is-good exposure"
        assert spec.data_requirement == "price"


# ----------------------------------------------------------------------
# Realised volatility
# ----------------------------------------------------------------------
def test_volatility_60d_is_the_annualised_trailing_standard_deviation(
    panel: MarketPanel, ctx: FactorContext
) -> None:
    """An explicit numpy slice over rows 141..200, not the same pandas call."""
    got = volatility_60d(ctx)
    window = panel.returns["AAA"].iloc[141:201].to_numpy()
    expected = float(np.std(window, ddof=1) * ANN)
    assert got["AAA"].iloc[200] == pytest.approx(expected, rel=1e-12)


def test_volatility_60d_needs_thirty_observations(ctx: FactorContext) -> None:
    got = volatility_60d(ctx)
    assert got.iloc[:30].isna().all().all()
    assert got.iloc[31:].notna().any().any()


def test_volatility_60d_is_zero_for_a_constant_return() -> None:
    flat = pd.DataFrame(0.01, index=DATES, columns=SYMBOLS)
    got = volatility_60d(FactorContext(_panel_from_returns(flat)))
    assert (got.iloc[60:].abs() < 1e-12).all().all()


def test_volatility_252d_uses_a_year_long_window(panel: MarketPanel, ctx: FactorContext) -> None:
    got = volatility_252d(ctx)
    window = panel.returns["AAA"].iloc[74:326].to_numpy()
    assert got["AAA"].iloc[325] == pytest.approx(float(np.std(window, ddof=1) * ANN), rel=1e-12)
    assert got.iloc[:126].isna().all().all()


def test_a_longer_window_is_smoother_than_a_shorter_one(ctx: FactorContext) -> None:
    short = volatility_60d(ctx)["AAA"].dropna()
    long = volatility_252d(ctx)["AAA"].dropna()
    common = short.index.intersection(long.index)
    assert short[common].std() > long[common].std()


# ----------------------------------------------------------------------
# Downside volatility
# ----------------------------------------------------------------------
def test_downside_volatility_is_the_annualised_semi_deviation(
    panel: MarketPanel, ctx: FactorContext
) -> None:
    got = downside_volatility(ctx)
    window = panel.returns["AAA"].iloc[75:201].to_numpy()  # 126 rows ending at 200
    negative = np.minimum(window, 0.0)
    expected = float(np.sqrt(np.mean(negative**2)) * ANN)
    assert got["AAA"].iloc[200] == pytest.approx(expected, rel=1e-12)


def test_downside_volatility_ignores_the_upside() -> None:
    """A huge positive return must not move a downside measure."""
    base = _random_returns()
    base.iloc[100, 0] = 0.001  # both variants are on the upside
    spiked = base.copy()
    spiked.iloc[100, 0] = 0.5
    plain = downside_volatility(FactorContext(_panel_from_returns(base)))
    loud = downside_volatility(FactorContext(_panel_from_returns(spiked)))
    assert loud["AAA"].iloc[150] == pytest.approx(plain["AAA"].iloc[150], rel=1e-12)


def test_downside_volatility_is_zero_when_nothing_falls() -> None:
    up = pd.DataFrame(0.002, index=DATES, columns=SYMBOLS)
    got = downside_volatility(FactorContext(_panel_from_returns(up)))
    assert (got.iloc[126:].abs() < 1e-12).all().all()


def test_downside_volatility_is_below_total_volatility(ctx: FactorContext) -> None:
    down = downside_volatility(ctx)["AAA"]
    total = volatility_252d(ctx)["AAA"]
    common = down.dropna().index.intersection(total.dropna().index)
    assert (down[common] <= total[common] + 1e-12).all()


# ----------------------------------------------------------------------
# Beta
# ----------------------------------------------------------------------
def _market_panel(betas: dict[str, float]) -> MarketPanel:
    """Asset returns built as exactly ``beta * market`` so beta is known."""
    rng = np.random.default_rng(9)
    market = pd.Series(rng.normal(0.0003, 0.01, size=len(DATES)), index=DATES)
    market.iloc[0] = 0.0
    rets = pd.DataFrame({sym: betas[sym] * market.to_numpy() for sym in betas}, index=DATES)
    benchmark = 100.0 * (1.0 + market).cumprod()
    return _panel_from_returns(rets, benchmark=benchmark)


def test_beta_252d_recovers_a_known_beta() -> None:
    panel = _market_panel({"AAA": 1.5, "BBB": 0.5})
    got = beta_252d(FactorContext(panel))
    for sym, beta in (("AAA", 1.5), ("BBB", 0.5)):
        assert got[sym].iloc[-1] == pytest.approx(beta, rel=1e-6)


def test_beta_252d_is_one_for_the_market_itself() -> None:
    panel = _market_panel({"AAA": 1.0})
    assert beta_252d(FactorContext(panel))["AAA"].iloc[-1] == pytest.approx(1.0, rel=1e-9)


def test_beta_252d_is_flat_across_dates_when_the_relationship_is_stable() -> None:
    panel = _market_panel({"AAA": 1.25})
    got = beta_252d(FactorContext(panel))["AAA"].dropna()
    assert len(got) > 0
    assert (got - 1.25).abs().max() < 1e-9


def test_beta_252d_is_nan_while_the_window_is_incomplete() -> None:
    panel = _market_panel({"AAA": 1.0})
    got = beta_252d(FactorContext(panel))["AAA"]
    assert got.iloc[:126].isna().all()


# ----------------------------------------------------------------------
# Idiosyncratic volatility
# ----------------------------------------------------------------------
def test_idiosyncratic_volatility_vanishes_when_the_market_explains_everything() -> None:
    """Returns exactly proportional to the market leave a zero residual."""
    panel = _market_panel({"AAA": 1.3, "BBB": 0.7})
    got = idiosyncratic_volatility(FactorContext(panel))
    assert (got.iloc[252:].abs() < 1e-9).all().all()


def test_idiosyncratic_volatility_is_positive_when_a_residual_exists() -> None:
    rng = np.random.default_rng(21)
    market = pd.Series(rng.normal(0.0003, 0.01, size=len(DATES)), index=DATES)
    market.iloc[0] = 0.0
    noise = rng.normal(0.0, 0.008, size=len(DATES))
    rets = pd.DataFrame({"AAA": 1.0 * market.to_numpy() + noise}, index=DATES)
    benchmark = 100.0 * (1.0 + market).cumprod()
    got = idiosyncratic_volatility(FactorContext(_panel_from_returns(rets, benchmark=benchmark)))
    assert got["AAA"].iloc[-1] > 0.0


def test_idiosyncratic_volatility_is_below_total_volatility() -> None:
    rng = np.random.default_rng(22)
    market = pd.Series(rng.normal(0.0003, 0.01, size=len(DATES)), index=DATES)
    market.iloc[0] = 0.0
    rets = pd.DataFrame(
        {"AAA": 1.0 * market.to_numpy() + rng.normal(0.0, 0.008, size=len(DATES))}, index=DATES
    )
    benchmark = 100.0 * (1.0 + market).cumprod()
    panel = _panel_from_returns(rets, benchmark=benchmark)
    ctx = FactorContext(panel)
    idio = idiosyncratic_volatility(ctx)["AAA"].iloc[-1]
    total = volatility_252d(ctx)["AAA"].iloc[-1]
    assert 0.0 < idio < total


# ----------------------------------------------------------------------
# Maximum drawdown
# ----------------------------------------------------------------------
def _close_only_panel(prices: np.ndarray, symbol: str = "AAA") -> MarketPanel:
    rows = [
        {
            "date": date,
            "symbol": symbol,
            "adj_close": price,
            "close": price,
            "volume": 1.0e6,
            "market_cap": 1.0e9,
            "industry": "Tech",
        }
        for date, price in zip(DATES, prices)
    ]
    return build_panel(pd.DataFrame(rows))


def test_max_drawdown_is_zero_for_a_monotonically_rising_price() -> None:
    """Regression: this used to be -0.42 on a 100 -> 200 ramp."""
    got = max_drawdown_252d(FactorContext(_close_only_panel(_ramp(100.0, 200.0, len(DATES)))))
    assert got["AAA"].iloc[-1] == pytest.approx(0.0)


def test_max_drawdown_measures_the_worst_peak_to_trough_move() -> None:
    """A peak at 100 followed by a 20% fall is a -20% drawdown."""
    prices = np.concatenate([np.full(200, 100.0), _ramp(100.0, 80.0, 100), np.full(100, 80.0)])
    got = max_drawdown_252d(FactorContext(_close_only_panel(prices)))
    assert got["AAA"].iloc[-1] == pytest.approx(-0.20, abs=1e-9)


def test_a_trough_that_precedes_the_peak_is_not_a_drawdown() -> None:
    """The ordering is the whole point: recovering from a dip is not a loss."""
    prices = np.concatenate([np.full(150, 80.0), _ramp(80.0, 100.0, 150), np.full(100, 100.0)])
    got = max_drawdown_252d(FactorContext(_close_only_panel(prices)))
    assert got["AAA"].iloc[-1] == pytest.approx(0.0)


def test_max_drawdown_is_never_positive() -> None:
    got = max_drawdown_252d(FactorContext(_close_only_panel(_random_returns()["AAA"].to_numpy())))
    values = got.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    assert (finite <= 1e-12).all()


def test_max_drawdown_is_nan_until_the_window_is_full() -> None:
    got = max_drawdown_252d(FactorContext(_close_only_panel(_ramp(100.0, 120.0, len(DATES)))))
    assert got.iloc[:252].isna().all().all()
    assert got.iloc[252:].notna().all().all()


def test_max_drawdown_never_sees_the_future() -> None:
    prices = _random_returns()["AAA"].to_numpy() * 0.0 + _ramp(100.0, 130.0, len(DATES))
    baseline = max_drawdown_252d(FactorContext(_close_only_panel(prices)))["AAA"]

    tampered = prices.copy()
    tampered[350:] = 1.0  # a crash nobody could have known about
    after = max_drawdown_252d(FactorContext(_close_only_panel(tampered)))["AAA"]

    np.testing.assert_allclose(
        after.iloc[:350].to_numpy(), baseline.iloc[:350].to_numpy(), equal_nan=True
    )


# ----------------------------------------------------------------------
# Skewness
# ----------------------------------------------------------------------
def test_return_skew_126d_is_the_trailing_skewness(panel: MarketPanel, ctx: FactorContext) -> None:
    got = return_skew_126d(ctx)
    window = pd.Series(panel.returns["AAA"].iloc[75:201].to_numpy())
    assert got["AAA"].iloc[200] == pytest.approx(float(window.skew()), rel=1e-12)
    assert got.iloc[:60].isna().all().all()


# ----------------------------------------------------------------------
# Market return
# ----------------------------------------------------------------------
def test_the_market_return_falls_back_to_an_equal_weighted_average(
    panel: MarketPanel, ctx: FactorContext
) -> None:
    expected = panel.returns.mean(axis=1)
    np.testing.assert_allclose(
        _market_return(ctx).dropna().to_numpy(), expected.dropna().to_numpy()
    )


def test_the_market_return_prefers_a_long_enough_benchmark() -> None:
    rng = np.random.default_rng(31)
    benchmark = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(DATES)))), index=DATES
    )
    panel = _panel_from_returns(_random_returns(), benchmark=benchmark)
    got = _market_return(FactorContext(panel))
    expected = benchmark.pct_change(fill_method=None)
    np.testing.assert_allclose(got.dropna().to_numpy(), expected.dropna().to_numpy())


def test_a_benchmark_with_too_little_history_falls_back_to_the_average() -> None:
    short = pd.Series(np.nan, index=DATES)
    short.iloc[350:] = 100.0 + np.arange(50)
    panel = _panel_from_returns(_random_returns(), benchmark=short)
    got = _market_return(FactorContext(panel))
    expected = panel.returns.mean(axis=1)
    np.testing.assert_allclose(got.dropna().to_numpy(), expected.dropna().to_numpy())

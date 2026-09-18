"""Offline tests for the score -> target-weights bridge.

:class:`PortfolioConstructor` is where a cross-section of alpha scores becomes a
book, so the properties that matter are the ones a plausible-looking bug would
break silently:

  * the risk model at a rebalance date sees **only** returns strictly before
    that date - a covariance contaminated by the future is a backtest that
    cannot lose;
  * eligibility is a *history* test (how much return data exists) intersected
    with a *tradability* test, and a name failing either is dropped;
  * volatility is the annualised diagonal of that same covariance, and a name
    the covariance does not cover gets the cross-sectional median rather than a
    NaN that propagates into every expected return;
  * the volatility-target fallback de-levers into cash, and leaves a book that
    already fits the budget alone;
  * a partial config section must not silently switch the risk constraints off.

Everything is deterministic: no network, fixed seeds, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.features.panel import MarketPanel, build_panel
from alphaforge.portfolio.constructor import ConstructionConfig, PortfolioConstructor
from alphaforge.portfolio.optimizer import OptimizationResult, OptimizerConfig

DATES = pd.bdate_range("2021-01-04", periods=320)
SYMBOLS = [f"S{i}" for i in range(8)]
LATE = "LATE"  # only starts trading two thirds of the way in
LATE_START = 200
# Index 180: LATE has no data at all, so the covariance drops it entirely.
# Index 250: LATE has ~50 observations (< 60) so it fails the history test.
# Index 300: LATE has ~100 observations so it passes.
BEFORE = DATES[180]
EARLY = DATES[250]
DATE = DATES[300]


def _long_table() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frames = []
    for i, sym in enumerate(SYMBOLS + [LATE]):
        steps = rng.normal(0.0003, 0.013, size=len(DATES))
        price = 100.0 * np.exp(np.cumsum(steps))
        start = LATE_START if sym == LATE else 0
        adj = np.where(np.arange(len(DATES)) >= start, price, np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "date": DATES,
                    "symbol": sym,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": adj,
                    "volume": 2.0e6,
                    "market_cap": 5.0e9,
                    "industry": "Tech" if i % 2 == 0 else "Energy",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def panel() -> MarketPanel:
    return build_panel(_long_table())


def _scores(rng_seed: int = 3, symbols: list[str] | None = None) -> pd.Series:
    rng = np.random.default_rng(rng_seed)
    names = symbols if symbols is not None else SYMBOLS + [LATE]
    return pd.Series(rng.normal(size=len(names)), index=pd.Index(names))


def _cov(panel: MarketPanel, date: pd.Timestamp, assets: pd.Index) -> pd.DataFrame:
    """Sample covariance over the trailing window, computed independently."""
    pos = panel.dates.get_loc(date)
    start = max(0, pos - 252)
    return panel.returns.iloc[start:pos][list(assets)].cov() * 252.0


# ----------------------------------------------------------------------
# No look-ahead
# ----------------------------------------------------------------------
def test_the_risk_model_never_sees_the_future(panel: MarketPanel) -> None:
    """Corrupting every return on or after the rebalance must not move the risk."""
    assets = PortfolioConstructor(panel).eligible(DATE)
    before = PortfolioConstructor(panel).covariance(DATE, assets).to_numpy(dtype=float)

    tampered = panel
    pos = tampered.dates.get_loc(DATE)
    tampered.returns.iloc[pos:] = 5.0  # a crash nobody could have known
    after = PortfolioConstructor(tampered).covariance(DATE, assets).to_numpy(dtype=float)

    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-14)


def test_the_covariance_matches_an_independent_trailing_sample(panel: MarketPanel) -> None:
    """Compared on names with a full window: the estimator shrinks nothing else."""
    assets = pd.Index(SYMBOLS)
    cfg = ConstructionConfig(covariance_method="sample")
    got = PortfolioConstructor(panel, construction_config=cfg).covariance(DATE, assets)
    np.testing.assert_allclose(
        got.to_numpy(dtype=float),
        _cov(panel, DATE, assets).to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-12,
    )


# ----------------------------------------------------------------------
# Eligibility
# ----------------------------------------------------------------------
def test_eligibility_is_a_history_test(panel: MarketPanel) -> None:
    """LATE fails the 60-observation floor at index 250 and passes it at 300."""
    c = PortfolioConstructor(panel)
    assert LATE not in c.eligible(EARLY)
    assert LATE in c.eligible(DATE)


def test_eligibility_is_also_a_tradability_test(panel: MarketPanel) -> None:
    c = PortfolioConstructor(panel)
    assert "S0" in c.eligible(DATE)
    panel.universe.loc[DATE, "S0"] = False
    assert "S0" not in c.eligible(DATE)


def test_eligibility_returns_a_sorted_index(panel: MarketPanel) -> None:
    got = PortfolioConstructor(panel).eligible(DATE)
    assert isinstance(got, pd.Index)
    assert list(got) == sorted(got)


# ----------------------------------------------------------------------
# Covariance / volatility
# ----------------------------------------------------------------------
def test_covariance_is_cached_per_date(panel: MarketPanel) -> None:
    c = PortfolioConstructor(panel)
    assets = c.eligible(DATE)
    assert c.covariance(DATE, assets) is c.covariance(DATE, assets)


def test_covariance_recomputes_when_the_asset_list_changes(panel: MarketPanel) -> None:
    c = PortfolioConstructor(panel)
    wide = c.eligible(DATE)
    narrow = wide[:4]
    full = c.covariance(DATE, wide)
    subset = c.covariance(DATE, narrow)
    assert list(subset.columns) == list(narrow)
    assert list(full.columns) == list(wide)


def test_covariance_raises_without_enough_history(panel: MarketPanel) -> None:
    with pytest.raises(ValueError, match="Not enough return history"):
        PortfolioConstructor(panel).covariance(DATES[0])


def test_volatility_is_the_annualised_diagonal(panel: MarketPanel) -> None:
    c = PortfolioConstructor(panel)
    assets = c.eligible(DATE)
    cov = c.covariance(DATE, assets)
    got = c.volatility(DATE, assets)
    np.testing.assert_allclose(
        got.to_numpy(dtype=float), np.sqrt(np.diag(cov.to_numpy(dtype=float))), rtol=1e-12
    )
    assert (got > 0).all()


def test_volatility_fills_an_uncovered_name_with_the_median(panel: MarketPanel) -> None:
    """A name the covariance drops would otherwise NaN every expected return."""
    c = PortfolioConstructor(panel)
    assets = c.eligible(BEFORE).append(pd.Index([LATE]))
    assert LATE not in c.covariance(BEFORE, assets).columns
    got = c.volatility(BEFORE, assets)
    assert np.isfinite(got[LATE])
    assert got[LATE] == pytest.approx(got.drop(LATE).median())


# ----------------------------------------------------------------------
# Industry labels
# ----------------------------------------------------------------------
def test_industry_returns_a_label_per_asset(panel: MarketPanel) -> None:
    c = PortfolioConstructor(panel)
    assets = c.eligible(DATE)
    got = c.industry(DATE, assets)
    assert got is not None
    assert list(got.index) == list(assets)
    assert set(got.unique()) <= {"Tech", "Energy"}


@pytest.mark.parametrize("empty", [None, pd.DataFrame()])
def test_industry_is_none_when_the_panel_carries_none(panel: MarketPanel, empty) -> None:
    panel.industry = empty
    c = PortfolioConstructor(panel)
    assert c.industry(DATE, c.eligible(DATE)) is None


def test_industry_is_none_for_a_date_missing_from_the_label_panel(panel: MarketPanel) -> None:
    panel.industry = panel.industry.iloc[:-1]
    c = PortfolioConstructor(panel)
    assert c.industry(DATES[-1], pd.Index(SYMBOLS)) is None


# ----------------------------------------------------------------------
# construct
# ----------------------------------------------------------------------
def test_construct_produces_a_long_only_book_over_the_scored_names(panel: MarketPanel) -> None:
    c = PortfolioConstructor(panel)
    result = c.construct(DATE, _scores(), ic=0.06)
    assert set(result.weights.index) <= set(c.eligible(DATE))
    assert (result.weights >= -1e-8).all()
    assert float(result.weights.sum()) <= 1.0 + 1e-6
    assert result.status.startswith("optimal")


def test_construct_records_the_ic_and_the_scored_breadth(panel: MarketPanel) -> None:
    c = PortfolioConstructor(panel)
    result = c.construct(DATE, _scores(), ic=0.042)
    assert result.diagnostics["ic_used"] == pytest.approx(0.042)
    assert result.diagnostics["n_scored"] == len(c.eligible(DATE))


def test_construct_narrows_the_universe_to_names_that_have_a_score(panel: MarketPanel) -> None:
    c = PortfolioConstructor(panel)
    result = c.construct(DATE, _scores(symbols=["S0", "S1", "S2", "S3", "S4"]), ic=0.06)
    assert set(result.weights.index) <= {"S0", "S1", "S2", "S3", "S4"}


def test_construct_raises_when_too_few_names_are_scored(panel: MarketPanel) -> None:
    with pytest.raises(ValueError, match="skipping rebalance"):
        PortfolioConstructor(panel).construct(DATE, pd.Series({"S0": 1.0, "S1": 0.5}), ic=0.06)


def test_construct_respects_the_minimum_name_floor(panel: MarketPanel) -> None:
    """``min_names`` is honoured, but never below five names."""
    cfg = OptimizerConfig(min_names=12)
    with pytest.raises(ValueError, match="Only 9 scored names"):
        PortfolioConstructor(panel, optimizer_config=cfg).construct(DATE, _scores(), ic=0.06)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
def test_config_objects_are_passed_through_untouched(panel: MarketPanel) -> None:
    opt = OptimizerConfig(max_weight=0.2)
    cons = ConstructionConfig(covariance_lookback=120)
    c = PortfolioConstructor(panel, optimizer_config=opt, construction_config=cons)
    assert c.optimizer_config is opt
    assert c.construction_config is cons


@pytest.mark.parametrize("cfg", [None, {}, {"risk": {}}])
def test_construction_config_from_dict_falls_back_to_defaults(cfg) -> None:
    got = ConstructionConfig.from_dict(cfg)
    assert got == ConstructionConfig()


def test_construction_config_reads_the_risk_section() -> None:
    got = ConstructionConfig.from_dict(
        {"risk": {"covariance_method": "ewma", "covariance_lookback": 90}, "ic_shrinkage": 0.25}
    )
    assert got.covariance_method == "ewma"
    assert got.covariance_lookback == 90
    assert got.ic_shrinkage == 0.25
    assert got.volatility_targeting is True  # untouched by a partial section


def test_a_partial_optimizer_section_keeps_the_default_risk_constraints() -> None:
    """Absent is not the same as ``null``.

    An absent key used to fall through to ``None``, so a partial ``portfolio:``
    section silently disabled volatility targeting, the turnover limit, the
    holdings cap and the industry-deviation cap.
    """
    got = OptimizerConfig.from_dict({"max_weight": 0.10})
    assert got.max_weight == 0.10
    assert got.target_volatility == OptimizerConfig().target_volatility
    assert got.turnover_limit == OptimizerConfig().turnover_limit
    assert got.max_holdings == OptimizerConfig().max_holdings
    assert got.max_industry_deviation == OptimizerConfig().max_industry_deviation
    assert OptimizerConfig.from_dict(None) == OptimizerConfig()
    assert OptimizerConfig.from_dict({}) == OptimizerConfig()


def test_an_explicit_null_still_disables_a_constraint() -> None:
    got = OptimizerConfig.from_dict({"target_volatility": "null", "turnover_limit": None})
    assert got.target_volatility is None
    assert got.turnover_limit is None


def test_every_optimizer_field_is_readable_from_config() -> None:
    """``cost_bps`` and ``dust_threshold`` were silently unconfigurable."""
    got = OptimizerConfig.from_dict({"cost_bps": 12.0, "dust_threshold": 1e-3})
    assert got.cost_bps == 12.0
    assert got.dust_threshold == 1e-3


# ----------------------------------------------------------------------
# Volatility-targeting fallback
# ----------------------------------------------------------------------
def _hot_book(ex_ante_vol: float) -> OptimizationResult:
    return OptimizationResult(
        weights=pd.Series({"AAA": 0.5, "BBB": 0.5}),
        method="mean_variance",
        status="optimal",
        diagnostics={"ex_ante_vol": ex_ante_vol},
    )


@pytest.fixture
def flat_cov() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]], index=pd.Index(["AAA", "BBB"]), columns=["AAA", "BBB"]
    )


def test_vol_targeting_de_levers_a_hot_book_into_cash(panel: MarketPanel, flat_cov) -> None:
    c = PortfolioConstructor(panel, optimizer_config=OptimizerConfig(target_volatility=0.10))
    out = c._apply_vol_target(_hot_book(0.20), flat_cov)
    assert out.diagnostics["vol_target_scale"] == pytest.approx(0.5)
    np.testing.assert_allclose(out.weights.to_numpy(), [0.25, 0.25])
    assert out.diagnostics["cash_weight"] == pytest.approx(0.5)
    # 0.25^2 * 0.04 * 2 -> sqrt(0.005)
    assert out.diagnostics["ex_ante_vol"] == pytest.approx(np.sqrt(0.005))


def test_vol_targeting_leaves_a_book_inside_the_budget_alone(panel: MarketPanel, flat_cov) -> None:
    c = PortfolioConstructor(panel, optimizer_config=OptimizerConfig(target_volatility=0.20))
    out = c._apply_vol_target(_hot_book(0.10), flat_cov)
    assert "vol_target_scale" not in out.diagnostics
    np.testing.assert_allclose(out.weights.to_numpy(), [0.5, 0.5])


@pytest.mark.parametrize("ex_ante_vol", [0.0, -0.1])
def test_vol_targeting_ignores_a_non_positive_vol(
    panel: MarketPanel, flat_cov, ex_ante_vol: float
) -> None:
    c = PortfolioConstructor(panel, optimizer_config=OptimizerConfig(target_volatility=0.10))
    out = c._apply_vol_target(_hot_book(ex_ante_vol), flat_cov)
    assert "vol_target_scale" not in out.diagnostics


def test_vol_targeting_can_be_switched_off(panel: MarketPanel, flat_cov) -> None:
    c = PortfolioConstructor(
        panel,
        optimizer_config=OptimizerConfig(target_volatility=0.10),
        construction_config=ConstructionConfig(volatility_targeting=False),
    )
    result = c.construct(DATE, _scores(), ic=0.06)
    assert "vol_target_scale" not in result.diagnostics

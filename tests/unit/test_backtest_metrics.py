"""Unit tests for the Brinson attribution and backtest performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.attribution.brinson import brinson_attribution
from alphaforge.backtest.metrics import (
    MetricsConfig,
    _drawdown_duration,
    _max_drawdown,
    drawdown_table,
    gross_returns_from_net,
    monthly_returns,
    performance_stats,
    rolling_metrics,
    summarise,
    yearly_returns,
)


def test_performance_stats_basic():
    rng = np.random.default_rng(0)
    rets = pd.Series(
        rng.normal(0.0004, 0.01, size=504), index=pd.date_range("2021-01-01", periods=504, freq="B")
    )
    m = performance_stats(rets)
    assert np.isfinite(m["cagr"])
    assert np.isfinite(m["sharpe"])
    assert m["max_drawdown"] <= 0.0
    assert 200 < m["periods_per_year"] < 300  # inference fix


def test_performance_stats_with_benchmark():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2021-01-01", periods=504, freq="B")
    strat = pd.Series(rng.normal(0.0005, 0.01, size=504), index=idx)
    bench = pd.Series(rng.normal(0.0003, 0.01, size=504), index=idx)
    m = performance_stats(strat, benchmark=bench)
    assert np.isfinite(m["information_ratio"])
    assert np.isfinite(m["alpha_ann"])


def test_active_return_is_terminal_wealth_gap_not_compounded_spread():
    """Active return must be the gap between two compounded curves.

    Regression guard. The previous implementation returned
    ``compound(s - b).iloc[-1] - 1``, i.e. it compounded the arithmetic return
    *spread* as if it were a return. That is meaningless economically and
    numerically explosive: any day with ``s - b < -1`` makes the factor
    ``1 + (s - b)`` negative with magnitude > 1, so a few thousand sessions of
    ``cumprod`` overflow float64 and every reported statistic becomes nan.
    """
    idx = pd.date_range("2021-01-01", periods=252, freq="B")
    rng = np.random.default_rng(3)
    strat = pd.Series(rng.normal(0.0006, 0.011, size=252), index=idx)
    bench = pd.Series(rng.normal(0.0003, 0.012, size=252), index=idx)

    m = performance_stats(strat, benchmark=bench)

    # Align on the overlapping window the stats are actually computed over.
    joined = pd.concat([strat.rename("s"), bench.rename("b")], axis=1).dropna()
    expected = float((1 + joined["s"]).prod() - (1 + joined["b"]).prod())

    assert np.isfinite(m["active_return"]), "active_return must never be nan/inf"
    assert abs(m["active_return"] - expected) < 1e-9


def test_relative_stats_finite_under_large_return_spread():
    """A day where the strategy badly lags the benchmark must not blow up."""
    idx = pd.date_range("2021-01-01", periods=500, freq="B")
    rng = np.random.default_rng(5)
    strat = pd.Series(rng.normal(0.0, 0.01, size=500), index=idx)
    bench = pd.Series(rng.normal(0.0, 0.01, size=500), index=idx)
    # Inject a spread far below -1 on a few days: the exact condition that used
    # to overflow the cumulative product.
    strat.iloc[100] = -0.90
    bench.iloc[100] = 2.50
    strat.iloc[250] = -0.95
    bench.iloc[250] = 1.80

    m = performance_stats(strat, benchmark=bench)
    for key in ("active_return", "information_ratio", "tracking_error", "beta", "alpha_ann"):
        assert np.isfinite(m[key]), f"{key} must stay finite, got {m[key]}"


def test_brinson_identity():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2022-01-01", periods=120, freq="B")
    symbols = [f"S{i}" for i in range(8)]
    asset_ret = pd.DataFrame(rng.normal(0.0, 0.01, size=(120, 8)), index=dates, columns=symbols)
    wp = pd.DataFrame(np.abs(rng.normal(0, 1, size=(120, 8))), index=dates, columns=symbols)
    wp = wp.div(wp.sum(axis=1), axis=0)
    wb = pd.DataFrame(np.abs(rng.normal(0, 1, size=(120, 8))), index=dates, columns=symbols)
    wb = wb.div(wb.sum(axis=1), axis=0)
    sectors = pd.Series(["X", "X", "Y", "Y", "Z", "Z", "X", "Y"], index=symbols)

    res = brinson_attribution(wp, wb, asset_ret, sectors)
    total = res.allocation + res.selection + res.interaction
    # The three-term Brinson-Fachler split is an *approximation*: it does not sum
    # to the realised active return exactly (the exact equality belongs to the
    # two-term Brinson-Hood form). The gap must stay small in absolute terms.
    assert abs(total - res.total_active) < 1e-3
    assert res.by_sector is not None
    assert "sector" in res.by_sector.columns


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def test_metrics_config_defaults():
    cfg = MetricsConfig()
    assert cfg.risk_free_rate == 0.0
    assert cfg.periods_per_year is None  # inferred from the index
    assert cfg.var_level == 0.95


def test_metrics_config_from_dict_reads_every_field():
    cfg = MetricsConfig.from_dict(
        {"risk_free_rate": 0.02, "periods_per_year": 12, "var_level": 0.99, "rolling_window": 60}
    )
    assert cfg.risk_free_rate == 0.02
    assert cfg.periods_per_year == 12
    assert cfg.var_level == 0.99
    assert cfg.rolling_window == 60


def test_a_null_periods_per_year_means_infer_it():
    """YAML writes `null`; the string form has to mean the same as absent."""
    assert MetricsConfig.from_dict({"periods_per_year": None}).periods_per_year is None
    assert MetricsConfig.from_dict({"periods_per_year": "null"}).periods_per_year is None


def test_an_empty_metrics_config_is_the_default():
    assert MetricsConfig.from_dict(None) == MetricsConfig()
    assert MetricsConfig.from_dict({}) == MetricsConfig()


# --------------------------------------------------------------------------
# performance_stats: independent recomputation
# --------------------------------------------------------------------------
def _monthly(seed: int = 7, n: int = 60) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.006, 0.03, size=n), index=pd.date_range("2018-01-31", periods=n, freq="ME")
    )


def test_the_headline_stats_match_an_independent_computation():
    """Sharpe, Sortino and vol recomputed from their definitions, not the code."""
    r = _monthly()
    cfg = MetricsConfig(periods_per_year=12, risk_free_rate=0.0)
    got = performance_stats(r, config=cfg)

    ppy = 12.0
    rf_period = (1.0 + 0.0) ** (1.0 / ppy) - 1.0
    excess = r - rf_period
    vol = float(r.std(ddof=1) * np.sqrt(ppy))
    downside = float(np.sqrt(np.mean(np.square(excess.clip(upper=0.0)))) * np.sqrt(ppy))

    assert got["ann_vol"] == pytest.approx(vol, rel=1e-12)
    assert got["sharpe"] == pytest.approx(float(excess.mean()) * ppy / vol, rel=1e-12)
    assert got["sortino"] == pytest.approx(float(excess.mean()) * ppy / downside, rel=1e-12)
    assert got["n_periods"] == len(r)
    assert got["total_return"] == pytest.approx(float(np.prod(1.0 + r) - 1.0), rel=1e-12)


def test_cagr_annualises_the_compounded_curve():
    r = _monthly(n=24)
    got = performance_stats(r, config=MetricsConfig(periods_per_year=12))
    curve = float(np.prod(1.0 + r))
    assert got["cagr"] == pytest.approx(curve ** (1.0 / 2.0) - 1.0, rel=1e-12)
    assert got["years"] == pytest.approx(2.0)


def test_var_is_the_tail_quantile_and_cvar_its_mean():
    r = _monthly()
    got = performance_stats(r, config=MetricsConfig(periods_per_year=12, var_level=0.95))
    var = float(np.quantile(r, 0.05))
    assert got["var_95"] == pytest.approx(var, rel=1e-12)
    assert got["cvar_95"] == pytest.approx(float(r[r <= var].mean()), rel=1e-12)
    assert got["cvar_95"] <= got["var_95"]


def test_an_empty_series_reports_an_error_instead_of_raising():
    got = performance_stats(pd.Series(dtype=float))
    assert got == {"error": "empty return series"}


def test_a_nan_only_series_is_treated_as_empty():
    got = performance_stats(
        pd.Series([np.nan, np.nan], index=pd.date_range("2020-01-01", periods=2))
    )
    assert got == {"error": "empty return series"}


def test_a_flat_series_has_no_drawdown_and_an_undefined_calmar():
    r = pd.Series(0.001, index=pd.date_range("2020-01-01", periods=252, freq="B"))
    got = performance_stats(r, config=MetricsConfig(periods_per_year=252))
    assert got["max_drawdown"] == pytest.approx(0.0)
    assert got["max_drawdown_duration_days"] == 0
    assert np.isnan(got["calmar"]), "no drawdown means calmar is undefined, not infinite"


def test_a_zero_variance_series_does_not_divide_by_zero():
    r = pd.Series(0.0, index=pd.date_range("2020-01-01", periods=100, freq="B"))
    got = performance_stats(r, config=MetricsConfig(periods_per_year=252))
    assert got["ann_vol"] == pytest.approx(0.0)
    assert not np.isfinite(got["sharpe"]) or got["sharpe"] == 0.0


def test_turnover_and_cost_drag_are_nan_when_not_supplied():
    got = performance_stats(_monthly())
    assert np.isnan(got["avg_turnover"])
    assert np.isnan(got["cost_drag_ann"])


def test_the_drawdown_date_is_the_trough():
    r = pd.Series([0.10, -0.30, 0.05], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    got = performance_stats(r, config=MetricsConfig(periods_per_year=12))
    assert got["max_drawdown"] == pytest.approx(-0.30, rel=1e-12)
    assert got["max_drawdown_date"] == "2020-02-29"


def test_relative_stats_need_at_least_three_overlapping_points():
    """Below three points beta and correlation are not defined; return nothing."""
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    r = pd.Series([0.01, 0.02], index=idx)
    bench = pd.Series([0.005, 0.01], index=idx)
    got = performance_stats(r, benchmark=bench, config=MetricsConfig(periods_per_year=12))
    assert "beta" not in got
    assert "benchmark_return" not in got


def test_beta_is_one_for_the_benchmark_itself():
    r = _monthly(seed=3, n=48)
    got = performance_stats(r, benchmark=r, config=MetricsConfig(periods_per_year=12))
    assert got["beta"] == pytest.approx(1.0, rel=1e-9)
    assert got["correlation"] == pytest.approx(1.0, rel=1e-9)
    assert got["r_squared"] == pytest.approx(1.0, rel=1e-9)
    assert got["active_return"] == pytest.approx(0.0, abs=1e-12)
    assert got["tracking_error"] == pytest.approx(0.0, abs=1e-12)


def test_capture_ratios_are_computed_on_the_right_benchmark_days():
    bench = pd.Series(
        [0.02, -0.02, 0.01, -0.01], index=pd.date_range("2020-01-31", periods=4, freq="ME")
    )
    strat = pd.Series([0.01, -0.01, 0.005, -0.005], index=bench.index)
    got = performance_stats(strat, benchmark=bench, config=MetricsConfig(periods_per_year=12))
    assert got["up_capture"] == pytest.approx(0.5, rel=1e-9)
    assert got["down_capture"] == pytest.approx(0.5, rel=1e-9)


# --------------------------------------------------------------------------
# Drawdown helpers
# --------------------------------------------------------------------------
def test_drawdown_duration_is_the_longest_underwater_stretch():
    r = pd.Series(
        [0.10, -0.10, -0.10, 0.30, -0.05],
        index=pd.date_range("2020-01-31", periods=5, freq="ME"),
    )
    got = performance_stats(r, config=MetricsConfig(periods_per_year=12))
    # Two sessions below the first peak, then one below the new peak.
    assert got["max_drawdown_duration_days"] == 2


def test_drawdown_helpers_survive_an_empty_series():
    assert _max_drawdown(pd.Series(dtype=float)) == (0.0, None)
    assert _drawdown_duration(pd.Series(dtype=float)) == 0


# --------------------------------------------------------------------------
# gross_returns_from_net
# --------------------------------------------------------------------------
def _engine_accounting(gross: np.ndarray, costs: np.ndarray, init: float):
    """Replay the backtest engine's NAV recursion.

    The pre-cost NAV compounds from the **post-cost** prior NAV, which is what
    makes the additive add-back exact - a probe that compounds the gross curve
    independently from ``init`` will disagree and look like a bug.
    """
    equity = np.empty(len(gross))
    net = np.empty(len(gross))
    prev = init
    for d in range(len(gross)):
        pre = prev * (1.0 + gross[d])
        equity[d] = pre - costs[d]
        net[d] = equity[d] / prev - 1.0
        prev = equity[d]
    return equity, net


def test_gross_returns_from_net_recovers_the_gross_series_exactly():
    """The docstring claims "exactly"; this is the identity that makes it true."""
    idx = pd.date_range("2020-01-01", periods=1500, freq="B")
    rng = np.random.default_rng(4)
    gross = rng.normal(0.0004, 0.011, size=len(idx))
    costs = np.where(rng.random(len(idx)) < 0.05, 900.0, 0.0)
    init = 1_000_000.0
    equity, net = _engine_accounting(gross, costs, init)

    got = gross_returns_from_net(
        pd.Series(net, index=idx), pd.Series(costs, index=idx), pd.Series(equity, index=idx), init
    )
    np.testing.assert_allclose(got.to_numpy(), gross, rtol=0, atol=1e-15)


def test_the_gross_net_gap_reconciles_the_cost_drag():
    idx = pd.date_range("2020-01-01", periods=1500, freq="B")
    rng = np.random.default_rng(4)
    gross = rng.normal(0.0004, 0.011, size=len(idx))
    costs = np.where(rng.random(len(idx)) < 0.05, 900.0, 0.0)
    init = 1_000_000.0
    equity, net = _engine_accounting(gross, costs, init)

    got = gross_returns_from_net(
        pd.Series(net, index=idx), pd.Series(costs, index=idx), pd.Series(equity, index=idx), init
    )
    assert got.sum() > pd.Series(net, index=idx).sum(), "gross must beat net"


def test_a_day_with_no_cost_is_left_alone():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    net = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
    zero = pd.Series(0.0, index=idx)
    equity = pd.Series(1.0 * np.cumprod(1.0 + net), index=idx)
    got = gross_returns_from_net(net, zero, equity, 1.0)
    np.testing.assert_allclose(got.to_numpy(), net.to_numpy(), rtol=1e-12)


def test_a_sparse_cost_series_is_reindexed():
    """Costs arrive on rebalance days only; the gaps must read as zero."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    net = pd.Series(0.01, index=idx)
    equity = pd.Series(1.0 * np.cumprod(1.0 + net), index=idx)
    sparse = pd.Series([0.0, 0.0, 100.0], index=idx[:3])
    got = gross_returns_from_net(net, sparse, equity, 1.0)
    assert got.notna().all()
    assert got.iloc[3] == pytest.approx(0.01, rel=1e-12)


# --------------------------------------------------------------------------
# Period tables
# --------------------------------------------------------------------------
def test_monthly_returns_compounds_within_each_month():
    r = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=pd.to_datetime(["2020-01-10", "2020-01-20", "2020-02-10", "2020-02-20"]),
    )
    got = monthly_returns(r)
    assert got.shape == (1, 2)
    assert list(got.columns) == ["Jan", "Feb"]
    assert got.iloc[0, 0] == pytest.approx(1.01 * 1.02 - 1.0, rel=1e-12)
    assert got.iloc[0, 1] == pytest.approx(0.99 * 1.03 - 1.0, rel=1e-12)


def test_monthly_returns_spans_years():
    idx = pd.date_range("2019-01-31", periods=24, freq="ME")
    got = monthly_returns(pd.Series(0.01, index=idx))
    assert list(got.index) == [2019, 2020]
    assert got.shape == (2, 12)


def test_monthly_returns_is_empty_for_an_empty_series():
    assert monthly_returns(pd.Series(dtype=float)).empty


def test_yearly_returns_compounds_within_each_year():
    idx = pd.date_range("2019-01-31", periods=24, freq="ME")
    got = yearly_returns(pd.Series(0.01, index=idx))
    assert list(got.index) == [2019, 2020]
    assert got.iloc[0] == pytest.approx(1.01**12 - 1.0, rel=1e-12)


def test_yearly_returns_is_empty_for_an_empty_series():
    got = yearly_returns(pd.Series(dtype=float))
    assert isinstance(got, pd.Series) and got.empty


# --------------------------------------------------------------------------
# rolling_metrics
# --------------------------------------------------------------------------
def test_rolling_metrics_columns_and_warmup():
    r = _monthly(n=48)
    got = rolling_metrics(r, window=12, periods_per_year=12)
    assert list(got.columns) == ["rolling_return", "rolling_vol", "rolling_sharpe", "drawdown"]
    assert got["rolling_vol"].iloc[:11].isna().all()
    assert got["rolling_vol"].iloc[11:].notna().all()


def test_rolling_return_is_the_trailing_compounded_return():
    r = pd.Series(0.01, index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    got = rolling_metrics(r, window=12, periods_per_year=12)
    assert got["rolling_return"].iloc[12] == pytest.approx(1.01**12 - 1.0, rel=1e-9)


def test_rolling_beta_appears_only_with_a_benchmark():
    r = _monthly(n=48)
    without = rolling_metrics(r, window=12, periods_per_year=12)
    assert "rolling_beta" not in without.columns
    with_bench = rolling_metrics(r, window=12, benchmark=r, periods_per_year=12)
    assert "rolling_beta" in with_bench.columns
    assert with_bench["rolling_beta"].iloc[-1] == pytest.approx(1.0, rel=1e-6)


def test_rolling_beta_waits_for_real_benchmark_observations():
    """A missing benchmark return must never be treated as a zero return."""
    idx = pd.date_range("2021-01-01", periods=6, freq="B")
    strategy = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.02], index=idx)
    benchmark = strategy.drop(idx[2])

    got = rolling_metrics(strategy, window=3, benchmark=benchmark, periods_per_year=252)

    assert got["rolling_beta"].iloc[:5].isna().all()
    assert got["rolling_beta"].iloc[-1] == pytest.approx(1.0, rel=1e-12)


# --------------------------------------------------------------------------
# drawdown_table
# --------------------------------------------------------------------------
def test_the_drawdown_table_lists_episodes_with_their_recovery():
    r = pd.Series(
        [0.10, -0.20, -0.10, 0.50, -0.10, 0.20],
        index=pd.date_range("2020-01-31", periods=6, freq="ME"),
    )
    got = drawdown_table(r, top=5)
    assert not got.empty
    assert list(got.columns) == [
        "start",
        "trough",
        "end",
        "depth",
        "length_days",
        "to_trough_days",
        "recovery_days",
    ]
    deepest = got.iloc[0]
    assert deepest["depth"] == pytest.approx(-0.28, rel=1e-9)  # 1.1*0.8*0.9 - 1
    assert deepest["start"] == "2020-02-29"
    assert deepest["trough"] == "2020-03-31"
    assert deepest["to_trough_days"] == 1


def test_the_drawdown_table_sorts_by_depth_and_truncates():
    rng = np.random.default_rng(2)
    r = pd.Series(
        rng.normal(-0.01, 0.05, size=240), index=pd.date_range("2000-01-31", periods=240, freq="ME")
    )
    got = drawdown_table(r, top=3)
    assert len(got) <= 3
    assert got["depth"].is_monotonic_increasing


def test_an_open_drawdown_is_still_reported():
    """A book that never recovers has no recovery date but is still an episode."""
    r = pd.Series([0.05, -0.10, -0.10], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    got = drawdown_table(r)
    assert len(got) == 1
    assert got.iloc[0]["end"] == "2020-03-31"


def test_the_drawdown_table_is_empty_without_drawdowns():
    r = pd.Series(0.01, index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    assert drawdown_table(r).empty


def test_the_drawdown_table_is_empty_for_an_empty_series():
    assert drawdown_table(pd.Series(dtype=float)).empty


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------
def test_summarise_is_a_key_value_frame():
    got = summarise({"sharpe": 1.2, "cagr": 0.08})
    assert list(got.columns) == ["metric", "value"]
    assert list(got["metric"]) == ["sharpe", "cagr"]
    assert list(got["value"]) == [1.2, 0.08]


def test_summarise_of_nothing_is_empty():
    assert summarise({}).empty

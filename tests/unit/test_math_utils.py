"""Unit tests for math / time utilities.

The numerical core is where silent damage happens: an off-by-one in
``forward_returns`` injects look-ahead bias into every factor label, a leak in
``expanding_window_index`` lets the future train the model, and a wrong IC
(spearman/pearson) makes a worthless factor look tradeable. Each of those
failure modes is asserted explicitly below rather than left to inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.utils.math_utils import (
    annualize_return,
    annualize_vol,
    compound,
    demean,
    drawdown_series,
    ensure_returns,
    expanding_window_index,
    first_valid_columns,
    forward_returns,
    infer_periods_per_year,
    max_drawdown,
    neutralize,
    pearson,
    rank_transform,
    safe_div,
    spearman,
    to_returns,
    winsorize,
    zscore,
)


def test_infer_periods_per_year_business_daily():
    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    ppy = infer_periods_per_year(idx)
    # Business-daily ~252 trading days per year, not 365000 (the old bug).
    assert 200 < ppy < 300


def test_infer_periods_per_year_monthly():
    idx = pd.date_range("2020-01-01", periods=48, freq="ME")
    ppy = infer_periods_per_year(idx)
    assert 11 < ppy < 13


def test_demean_within_group():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(3, 4)), columns=["A", "B", "C", "D"])
    group = pd.Series({"A": "x", "B": "x", "C": "y", "D": "y"})
    out = demean(df, group=group, axis=1)
    # Each group's columns are mean-zero per row.
    assert np.allclose(out[["A", "B"]].mean(axis=1).to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(out[["C", "D"]].mean(axis=1).to_numpy(), 0.0, atol=1e-9)
    # Univariate demean (no group) leaves the mean at zero.
    flat = demean(df)
    assert np.allclose(flat.mean(axis=1).to_numpy(), 0.0, atol=1e-9)


def test_demean_returns_frame_of_correct_shape():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(5, 3)), columns=list("abc"))
    out = demean(df, group=pd.Series({"a": "g", "b": "g", "c": "h"}), axis=1)
    assert out.shape == df.shape
    assert np.isfinite(out.to_numpy()).all()


# --------------------------------------------------------------------------
# compound() - numerical stability
# --------------------------------------------------------------------------
def test_compound_matches_cumprod_on_normal_returns():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0004, 0.01, size=2520))
    assert np.allclose(compound(r).to_numpy(), (1.0 + r).cumprod().to_numpy(), rtol=1e-9)


def test_compound_survives_long_horizon_without_overflow():
    """A 20-year daily series must stay finite.

    The regression this locks down: ``(1 + r).cumprod()`` overflowed/underflowed
    once the product left float64 range, turning every downstream statistic
    (CAGR, max drawdown, Sharpe) into nan.
    """
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(-0.02, 0.02, size=5040))  # 20y of persistent decay
    curve = compound(r)
    assert np.isfinite(curve.to_numpy()).all(), "compound() must not produce inf/nan"
    assert curve.iloc[0] > 0.0


def test_compound_handles_total_wipeout():
    """A -100% day zeroes the curve instead of producing nan."""
    r = pd.Series([0.10, -1.0, 0.05])
    curve = compound(r)
    assert np.isfinite(curve.to_numpy()).all()
    assert curve.iloc[-1] == 0.0


def test_compound_preserves_index_and_handles_nan():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    r = pd.Series([0.01, np.nan, 0.02, -0.01, 0.03], index=idx)
    curve = compound(r)
    assert list(curve.index) == list(idx)
    assert np.isfinite(curve.to_numpy()).all()


# --------------------------------------------------------------------------
# forward_returns() - the factor label generator (look-ahead hazard)
# --------------------------------------------------------------------------
def test_forward_returns_aligns_to_decision_date():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]})
    fwd = forward_returns(prices, horizon=1)
    assert fwd["A"].iloc[0] == pytest.approx(0.10)
    assert fwd["A"].iloc[1] == pytest.approx(0.10)
    assert fwd["A"].iloc[2] == pytest.approx(0.10)


def test_forward_returns_has_no_lookahead():
    """The tail is unknowable and must be NaN - never zero-filled."""
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1, 146.41]})
    fwd = forward_returns(prices, horizon=2)
    assert fwd["A"].iloc[0] == pytest.approx(121.0 / 100.0 - 1.0)
    assert fwd["A"].iloc[2] == pytest.approx(146.41 / 121.0 - 1.0)
    # The final `horizon` rows have no realised future and must be missing.
    assert fwd["A"].iloc[-2:].isna().all()
    assert fwd["A"].iloc[:-2].notna().all()


def test_forward_returns_log_method():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    fwd = forward_returns(prices, horizon=1, method="log")
    assert fwd["A"].iloc[0] == pytest.approx(np.log(1.10))
    # log(1 + simple) is the identity linking the two methods.
    simple = forward_returns(prices, horizon=1)
    # equal_nan=True: the unknowable tail is NaN on both sides.
    assert np.allclose(fwd.to_numpy(), np.log1p(simple.to_numpy()), equal_nan=True)


# --------------------------------------------------------------------------
# to_returns()
# --------------------------------------------------------------------------
def test_to_returns_simple_and_log():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    simple = to_returns(prices)
    assert simple["A"].iloc[1] == pytest.approx(0.10)
    assert simple["A"].iloc[2] == pytest.approx(-0.10)
    logr = to_returns(prices, method="log")
    assert logr["A"].iloc[1] == pytest.approx(np.log(1.10))
    assert logr["A"].iloc[2] == pytest.approx(np.log(0.90))


# --------------------------------------------------------------------------
# spearman() / pearson() - the IC metrics
# --------------------------------------------------------------------------
def test_spearman_perfect_and_inverse():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_level_based():
    """A monotone non-linear transform keeps Spearman at 1 but moves Pearson."""
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a**2
    assert spearman(a, b) == pytest.approx(1.0)
    assert pearson(a, b) < 0.99


def test_ic_metrics_need_at_least_three_observations():
    a = pd.Series([1.0, 2.0])
    b = pd.Series([2.0, 4.0])
    assert np.isnan(spearman(a, b))
    assert np.isnan(pearson(a, b))


def test_ic_metrics_drop_infinities_and_nans():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, np.inf, np.nan])
    b = pd.Series([2.0, 4.0, 6.0, 8.0, 1.0, 1.0])
    # inf/NaN pairs are dropped, leaving a perfectly ranked cross-section.
    assert spearman(a, b) == pytest.approx(1.0)
    assert pearson(a, b) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# max_drawdown() / drawdown_series()
# --------------------------------------------------------------------------
def test_max_drawdown_recovers_known_trough():
    # curve = [1.10, 0.88, 0.924] -> worst drawdown -20% at the middle row.
    r = pd.Series([0.10, -0.20, 0.05])
    dd, trough = max_drawdown(r)
    assert dd == pytest.approx(-0.20)
    assert trough == 1


def test_max_drawdown_monotone_gains_and_empty():
    up = pd.Series([0.01, 0.02, 0.03])
    dd, trough = max_drawdown(up)
    assert dd == pytest.approx(0.0)
    assert trough == 0
    assert max_drawdown(pd.Series(dtype=float)) == (0.0, None)


def test_drawdown_series_is_non_positive_and_ends_at_zero_peak():
    r = pd.Series([0.10, -0.20, 0.50])
    dd = drawdown_series(r)
    assert (dd <= 1e-12).all()
    assert dd.iloc[-1] == pytest.approx(0.0)  # new high -> no drawdown


# --------------------------------------------------------------------------
# expanding_window_index() - walk-forward leakage guard
# --------------------------------------------------------------------------
def test_expanding_window_never_trains_on_the_future():
    pairs = list(expanding_window_index(10, min_obs=4))
    assert len(pairs) == 6  # test positions 4..9
    for train, pos in pairs:
        assert train.max() < pos, "train window must end strictly before test"
        assert len(train) == pos


def test_expanding_window_empty_when_min_obs_exceeds_n():
    assert list(expanding_window_index(3, min_obs=5)) == []


# --------------------------------------------------------------------------
# neutralize()
# --------------------------------------------------------------------------
def test_neutralize_removes_exposure_and_intercept():
    rng = np.random.default_rng(3)
    b = rng.normal(size=300)
    f = 1.0 + 2.0 * b + rng.normal(scale=0.01, size=300)
    resid = neutralize(pd.Series(f), pd.DataFrame({"b": b}))
    # With an intercept the residual must be mean-zero and orthogonal to b.
    assert abs(resid.mean()) < 0.01
    assert abs(pearson(resid, pd.Series(b))) < 0.05
    assert len(resid) == 300


def test_neutralize_returns_nan_when_underdetermined():
    factor = pd.Series([1.0, 2.0], index=["x", "y"])
    exposures = pd.DataFrame({"b": [0.5, 0.7]}, index=["x", "y"])
    out = neutralize(factor, exposures)
    assert out.isna().all()
    assert list(out.index) == ["x", "y"]  # index preserved


def test_neutralize_drops_infinite_rows():
    factor = pd.Series([1.0, 2.0, np.inf, 4.0, 5.0])
    exposures = pd.DataFrame({"b": [0.1, 0.2, 0.3, 0.4, 0.5]})
    out = neutralize(factor, exposures)
    assert np.isfinite(out.dropna().to_numpy()).all()
    # Row 2 carried the inf factor value: it is excluded, not silently fitted.
    assert pd.isna(out.iloc[2])
    assert out.iloc[3:].notna().all()  # the clean rows still get residuals


# --------------------------------------------------------------------------
# Cross-sectional transforms
# --------------------------------------------------------------------------
def test_winsorize_clips_without_dropping_rows():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, -100.0])
    out = winsorize(s, lower=0.1, upper=0.1)
    assert len(out) == len(s)
    assert out.max() <= s.quantile(0.9) + 1e-12
    assert out.min() >= s.quantile(0.1) - 1e-12


def test_winsorize_asymmetric_bounds_and_all_nan():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    out = winsorize(s, lower=0.0, upper=0.25)
    assert out.max() == pytest.approx(s.quantile(0.75))
    assert out.min() == pytest.approx(s.min())
    empty = pd.Series([np.nan, np.nan])
    assert winsorize(empty).isna().all()


def test_zscore_is_zero_mean_unit_std_per_row():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(size=(50, 12)))
    out = zscore(df, axis=1)
    assert np.allclose(out.mean(axis=1).to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(out.std(axis=1, ddof=1).to_numpy(), 1.0, atol=1e-9)
    colwise = zscore(df, axis=0)
    assert np.allclose(colwise.mean(axis=0).to_numpy(), 0.0, atol=1e-9)


def test_zscore_constant_slice_becomes_nan_not_inf():
    """A zero-sigma row must degrade to NaN, never to ±inf."""
    df = pd.DataFrame([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])
    out = zscore(df, axis=1)
    assert out.iloc[0].isna().all()
    assert np.isfinite(out.iloc[1].to_numpy()).all()


def test_rank_transform_preserves_nan_and_bounds():
    df = pd.DataFrame([[3.0, 1.0, 2.0, np.nan], [10.0, 30.0, 20.0, 5.0]])
    out = rank_transform(df, axis=1, pct=True)
    assert out.iloc[0].max() == pytest.approx(1.0)
    assert ((out.dropna(axis=1) > 0) & (out.dropna(axis=1) <= 1.0)).all().all()
    assert pd.isna(out.iloc[0].iloc[-1])  # NaN stays NaN (na_option="keep")


def test_demean_along_rows_groups_by_index_label():
    df = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3))
    group = pd.Series({0: "a", 1: "a", 2: "b", 3: "b"})
    out = demean(df, group=group, axis=0)
    assert np.allclose(out.loc[[0, 1]].mean().to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(out.loc[[2, 3]].mean().to_numpy(), 0.0, atol=1e-9)


# --------------------------------------------------------------------------
# Annualisation / misc helpers
# --------------------------------------------------------------------------
def test_annualize_return_and_vol_use_252_by_default():
    assert annualize_return(0.0004, 252) == pytest.approx(1.0004**252 - 1.0)
    assert annualize_vol(0.01, 252) == pytest.approx(0.01 * np.sqrt(252))


def test_infer_periods_per_year_guards():
    # Too few observations to infer anything -> fall back to the daily default.
    assert infer_periods_per_year(pd.DatetimeIndex(["2020-01-01", "2020-01-02"])) == 252
    # Zero median gap (duplicate timestamps) must not divide by zero.
    dupes = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-01"])
    assert infer_periods_per_year(dupes) == 252


def test_safe_div_guards_zero_nan_and_none():
    assert safe_div(1.0, 0.0, default=0.0) == 0.0
    assert np.isnan(safe_div(1.0, np.nan))
    assert safe_div(1.0, None, default=-1.0) == -1.0
    assert safe_div(1.0, 4.0) == pytest.approx(0.25)


def test_first_valid_columns_filters_by_coverage():
    df = pd.DataFrame({"full": [1.0, 2.0, 3.0], "sparse": [1.0, np.nan, np.nan]})
    assert first_valid_columns(df, min_periods=3) == ["full"]
    assert first_valid_columns(df, min_periods=1) == ["full", "sparse"]


def test_ensure_returns_leaves_return_series_untouched():
    """A daily-return benchmark must NOT be differenced a second time."""
    rets = pd.Series([0.001, -0.002, 0.003])
    out = ensure_returns(rets)
    pd.testing.assert_series_equal(out, rets.astype(float))


def test_ensure_returns_converts_price_levels():
    levels = pd.Series([100.0, 110.0, 121.0])
    out = ensure_returns(levels)
    pd.testing.assert_series_equal(out, pd.Series([0.10, 0.10], index=[1, 2]))


def test_ensure_returns_empty_is_noop():
    assert ensure_returns(pd.Series(dtype=float)).empty

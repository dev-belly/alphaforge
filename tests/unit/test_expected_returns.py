"""Offline tests for the expected-returns bridge (``portfolio/expected_returns.py``).

This module turns a raw model score into an annualised expected return via
Grinold's fundamental law (``mu = shrunk_IC * z * sigma``) and blends several
alpha sources.  It is pure numpy/pandas, so every contract is locked without a
network:

  * ``implied_expected_returns`` de-means and z-scores the scores, clips them,
    scales by a shrunk IC and by volatility, and finally subtracts the mean so
    the alphas are cash-neutral (sum to ~0);
  * the IC shrinkage and the volatility both enter *linearly*, so scaling
    either scales ``mu`` proportionally;
  * an additive shift in the raw scores is ignored (we de-mean first);
  * a far-outlying score is bounded by ``clip_sigma`` and cannot dominate;
  * missing volatility is filled with the median volatility, not dropped;
  * ``blend_expected_returns`` normalises weights by their L1 norm, refuses an
    empty input, and rescales the blend to unit z-score;
  * ``realised_forward_returns`` is the textbook forward-return diagnostic.

Deterministic inputs keep the numeric assertions stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.portfolio.expected_returns import (
    blend_expected_returns,
    implied_expected_returns,
    realised_forward_returns,
)


def _scores(values, idx=None):
    idx = idx or [f"A{i}" for i in range(len(values))]
    return pd.Series(values, index=idx, dtype="float64")


def _vol(values, idx=None):
    idx = idx or [f"A{i}" for i in range(len(values))]
    return pd.Series(values, index=idx, dtype="float64")


# --------------------------------------------------------------------------- #
# implied_expected_returns
# --------------------------------------------------------------------------- #
def test_implied_returns_are_cash_neutral():
    scores = _scores([1.0, 3.0, -2.0, 0.5, 4.0])
    vol = _vol([0.2, 0.4, 0.15, 0.3, 0.25])
    mu = implied_expected_returns(scores, vol, ic=0.1)
    assert abs(mu.sum()) < 1e-9


def test_implied_returns_scale_linearly_with_ic():
    scores = _scores([1.0, 3.0, -2.0])
    vol = _vol([0.2, 0.4, 0.15])
    mu1 = implied_expected_returns(scores, vol, ic=0.1, ic_shrinkage=1.0)
    mu2 = implied_expected_returns(scores, vol, ic=0.2, ic_shrinkage=1.0)
    pd.testing.assert_series_equal(mu2, 2.0 * mu1, check_names=False)


def test_implied_returns_scale_linearly_with_sigma():
    scores = _scores([1.0, 3.0, -2.0])
    vol1 = _vol([0.2, 0.4, 0.15])
    vol2 = _vol([0.4, 0.8, 0.30])  # exactly 2x vol1
    mu1 = implied_expected_returns(scores, vol1, ic=0.1)
    mu2 = implied_expected_returns(scores, vol2, ic=0.1)
    pd.testing.assert_series_equal(mu2, 2.0 * mu1, check_names=False)


def test_ic_shrinkage_halves_mu():
    scores = _scores([1.0, 3.0, -2.0])
    vol = _vol([0.2, 0.4, 0.15])
    mu_full = implied_expected_returns(scores, vol, ic=0.1, ic_shrinkage=1.0)
    mu_half = implied_expected_returns(scores, vol, ic=0.1, ic_shrinkage=0.5)
    pd.testing.assert_series_equal(mu_half, 0.5 * mu_full, check_names=False)


def test_additive_shift_in_scores_is_ignored():
    scores = _scores([1.0, 3.0, -2.0])
    shifted = _scores([11.0, 13.0, 8.0])  # +10 to every score
    vol = _vol([0.2, 0.4, 0.15])
    mu_a = implied_expected_returns(scores, vol, ic=0.1)
    mu_b = implied_expected_returns(shifted, vol, ic=0.1)
    pd.testing.assert_series_equal(mu_a, mu_b, check_names=False)


def test_clipping_reduces_outlier_influence():
    # Two equal anchors + one extreme outlier among three points => z_outlier > 0.5.
    scores = _scores([0.0, 0.0, 1e9])
    vol = _vol([0.2, 0.2, 0.2])  # constant vol isolates the score effect
    mu_lo = implied_expected_returns(scores, vol, ic=0.1, clip_sigma=0.5)
    mu_hi = implied_expected_returns(scores, vol, ic=0.1, clip_sigma=100.0)
    # The clipped outlier must exert strictly less influence than the unclipped one.
    assert abs(mu_lo.iloc[2]) < abs(mu_hi.iloc[2])
    # And it is bounded by the clip: |z| <= clip_sigma.
    assert abs(mu_lo.iloc[2]) <= 0.1 * 0.5 * 0.2 + 1e-9


def test_annualisation_scales_sigma_by_sqrt_periods():
    # Constant vol + zero-mean scores => cash-neutral step is a no-op, so the
    # daily mu is exactly the annual mu divided by sqrt(periods_per_year).
    scores = _scores([-1.0, 0.0, 1.0])  # already zero-mean
    vol = _vol([0.2, 0.2, 0.2])  # constant
    mu_ann = implied_expected_returns(scores, vol, ic=0.1, annualise=True, periods_per_year=252.0)
    mu_day = implied_expected_returns(scores, vol, ic=0.1, annualise=False, periods_per_year=252.0)
    expected = mu_ann / np.sqrt(252.0)
    pd.testing.assert_series_equal(mu_day, expected, check_names=False, rtol=1e-9)


def test_missing_volatility_filled_with_median():
    scores = _scores([1.0, 3.0, 5.0], idx=["A", "B", "C"])
    # Volatility only known for A and B; C must fall back to the median (0.25).
    vol = _vol([0.2, 0.3], idx=["A", "B"])
    mu = implied_expected_returns(scores, vol, ic=0.1, ic_shrinkage=1.0)
    # Full reconstruction: C's sigma is the median volatility, not its own.
    z = scores - scores.mean()
    z = z / z.std()
    sigma = vol.reindex(scores.index).fillna(vol.median())
    raw = 0.1 * z * sigma
    expected = raw - raw.mean()
    pd.testing.assert_series_equal(mu, expected, check_names=False, rtol=1e-9)
    # C used the median, A/B kept their own sigma.
    assert abs(sigma["C"] - 0.25) < 1e-12
    assert abs(sigma["A"] - 0.2) < 1e-12 and abs(sigma["B"] - 0.3) < 1e-12


def test_nan_score_yields_zero_mu():
    scores = _scores([1.0, np.nan, 3.0])
    vol = _vol([0.2, 0.4, 0.15])
    mu = implied_expected_returns(scores, vol, ic=0.1, ic_shrinkage=1.0)
    assert mu.notna().all()
    # Full reconstruction.
    z = scores - scores.mean()
    z = z / z.std()
    z = z.clip(-3.0, 3.0).fillna(0.0)  # mirrors the function's NaN->0 step
    sigma = vol.reindex(scores.index).fillna(vol.median())
    raw = 0.1 * z * sigma
    expected = raw - raw.mean()
    pd.testing.assert_series_equal(mu, expected, check_names=False, rtol=1e-9)
    # The NaN score has z=0, so its *raw* (pre cash-neutral) alpha is exactly 0.
    nan_idx = scores.index[1]
    assert raw.loc[nan_idx] == 0.0


def test_reindex_restores_original_order_not_volatility_order():
    # Scores supplied as [B, A]; volatility keyed [A, B]. Result must be ordered
    # like the scores and each asset must use ITS OWN volatility.
    scores = _scores([1.0, 3.0], idx=["B", "A"])
    vol = _vol([0.4, 0.2], idx=["A", "B"])  # A=0.2, B=0.4
    mu = implied_expected_returns(scores, vol, ic=0.1, ic_shrinkage=1.0)
    assert list(mu.index) == ["B", "A"]
    z = scores - scores.mean()
    z = z / z.std()
    sigma = vol.reindex(scores.index)  # aligns B->0.2, A->0.4 (by label)
    raw = 0.1 * z * sigma
    expected = raw - raw.mean()
    pd.testing.assert_series_equal(mu, expected, check_names=False, rtol=1e-9)
    # B (index 0) used its own vol 0.2, not A's 0.4.
    assert abs(sigma["B"] - 0.2) < 1e-12


def test_hand_computed_two_asset_anchor():
    # scores [1, 3], vol [0.2, 0.4], ic 0.1, shrinkage 1.0, annualised.
    scores = _scores([1.0, 3.0])
    vol = _vol([0.2, 0.4])
    mu = implied_expected_returns(scores, vol, ic=0.1, ic_shrinkage=1.0)
    z = scores - 2.0
    z = z / z.std()  # sample std of [-1, 1] is sqrt(2)
    raw = 0.1 * z * vol
    expected = raw - raw.mean()
    pd.testing.assert_series_equal(mu, expected, check_names=False, rtol=1e-9)


# --------------------------------------------------------------------------- #
# blend_expected_returns
# --------------------------------------------------------------------------- #
def test_blend_rejects_empty_input():
    with pytest.raises(ValueError):
        blend_expected_returns({})


def test_blend_equal_weights_is_unit_z_of_average():
    c1 = _scores([0.1, -0.2, 0.3])
    c2 = _scores([0.2, 0.1, -0.1])
    acc = blend_expected_returns({"x": c1, "y": c2})
    expected = (c1 + c2) / 2.0
    expected = expected / expected.std()
    pd.testing.assert_series_equal(acc, expected, check_names=False, rtol=1e-9)
    assert abs(acc.std() - 1.0) < 1e-9


def test_blend_single_component_is_unit_z():
    c = _scores([0.1, -0.2, 0.3, 0.05])
    acc = blend_expected_returns({"x": c})
    expected = c / c.std()
    pd.testing.assert_series_equal(acc, expected, check_names=False, rtol=1e-9)


def test_blend_weights_normalise_to_l1():
    c1 = _scores([0.1, -0.2, 0.3])
    c2 = _scores([0.2, 0.1, -0.1])
    acc = blend_expected_returns({"x": c1, "y": c2}, weights={"x": 2.0, "y": 1.0})
    expected = (2.0 * c1 + 1.0 * c2) / 3.0
    expected = expected / expected.std()
    pd.testing.assert_series_equal(acc, expected, check_names=False, rtol=1e-9)


def test_blend_all_zero_weights_returns_zeros():
    c1 = _scores([0.1, -0.2, 0.3])
    c2 = _scores([0.2, 0.1, -0.1])
    acc = blend_expected_returns({"x": c1, "y": c2}, weights={"x": 0.0, "y": 0.0})
    assert (acc == 0.0).all()


def test_blend_handles_mismatched_indices():
    c1 = _scores([0.1, -0.2], idx=["A", "B"])
    c2 = _scores([0.3, 0.4], idx=["B", "C"])
    acc = blend_expected_returns({"x": c1, "y": c2})  # equal weights 1.0 each
    # Union index, missing components contribute 0 via fill_value.
    assert set(acc.index) == {"A", "B", "C"}
    # Pre-rescale weighted average: A=0.5*0.1, B=0.5*(-0.2)+0.5*0.3, C=0.5*0.4.
    pre = pd.Series({"A": 0.05, "B": 0.05, "C": 0.20})
    expected = pre / pre.std()
    pd.testing.assert_series_equal(acc, expected, check_names=False, rtol=1e-9)


# --------------------------------------------------------------------------- #
# realised_forward_returns (diagnostic)
# --------------------------------------------------------------------------- #
def test_forward_returns_basic():
    close = pd.DataFrame({"X": [100.0, 110.0, 121.0, 133.1]})
    fwd = realised_forward_returns(close, horizon=1)
    expected = close.shift(-1) / close - 1.0
    pd.testing.assert_frame_equal(fwd, expected, rtol=1e-9)
    assert fwd.iloc[-1].isna().all()  # last row has no forward horizon


def test_forward_returns_negative():
    close = pd.DataFrame({"X": [100.0, 90.0]})
    fwd = realised_forward_returns(close, horizon=1)
    assert abs(fwd.iloc[0]["X"] - (-0.1)) < 1e-9
    assert fwd.iloc[1].isna().all()


def test_forward_returns_uses_horizon():
    close = pd.DataFrame({"X": [100.0, 100.0, 100.0, 200.0]})
    fwd = realised_forward_returns(close, horizon=2)
    # shift(-2): index0 reads close[2]=100 -> 0.0 ; index1 reads close[3]=200 -> +1.0
    assert abs(fwd.iloc[0]["X"] - 0.0) < 1e-9
    assert abs(fwd.iloc[1]["X"] - 1.0) < 1e-9
    assert fwd.iloc[2].isna().all() and fwd.iloc[3].isna().all()

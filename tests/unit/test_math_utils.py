"""Unit tests for math / time utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.utils.math_utils import compound, demean, infer_periods_per_year


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

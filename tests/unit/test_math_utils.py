"""Unit tests for math / time utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.utils.math_utils import demean, infer_periods_per_year


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

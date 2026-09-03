"""Unit tests for the factor evaluation layer (IC, quantiles, decay, FDR)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.factors.evaluation import (
    benjamini_hochberg,
    compute_ic,
    factor_turnover,
    ic_decay,
    quantile_portfolios,
)


def _synth_factor_close(
    n: int = 400, k: int = 10, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A factor that is *predictive* of next-period returns + the close panel."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    syms = [f"S{i}" for i in range(k)]
    f = pd.DataFrame(rng.normal(0, 1, size=(n, k)), index=idx, columns=syms)
    # Forward return is weakly driven by the factor (positive IC) + noise.
    fwd = f * 0.02 + rng.normal(0, 0.02, size=(n, k))
    close = pd.DataFrame((1.0 + fwd.fillna(0.0)).cumprod(axis=0) * 100.0, index=idx, columns=syms)
    return f, close


def test_compute_ic_positive_and_finite():
    f, close = _synth_factor_close(seed=3)
    fwd = close.shift(-21) / close - 1.0
    ic = compute_ic(f, fwd)
    assert "pearson_ic" in ic.columns and "rank_ic" in ic.columns
    valid = ic.dropna()
    assert len(valid) >= 5
    assert valid["rank_ic"].std() > 0
    # The engineered factor has positive mean rank IC.
    assert valid["rank_ic"].mean() > 0


def test_compute_ic_respects_universe():
    f, close = _synth_factor_close(seed=4)
    fwd = close.shift(-21) / close - 1.0
    universe = pd.DataFrame(True, index=f.index, columns=f.columns)
    universe.iloc[:, :5] = False  # restrict to half the names
    ic_full = compute_ic(f, fwd)
    ic_univ = compute_ic(f, fwd, universe=universe)
    assert (ic_univ.notna().sum(axis=1) <= ic_full.notna().sum(axis=1) + 1).all()


def test_quantile_portfolios_spread_positive():
    f, close = _synth_factor_close(seed=5)
    fwd = close.shift(-21) / close - 1.0
    qrets, top = quantile_portfolios(f, fwd, n_quantiles=5)
    assert list(qrets.columns) == ["q1", "q2", "q3", "q4", "q5"]
    # Top quantile mean return should beat the bottom quantile on average.
    assert qrets["q5"].mean() > qrets["q1"].mean()


def test_factor_turnover_in_unit_interval():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-01-01", periods=100, freq="B")
    w = pd.DataFrame(rng.uniform(0, 1, size=(100, 8)), index=idx, columns=list("ABCDEFGH"))
    to = factor_turnover(w)
    assert ((to >= 0) & (to <= 2.0)).all()  # one-way turnover <= 2 by construction


def test_ic_decay_has_expected_horizons():
    f, close = _synth_factor_close(seed=8)
    decay = ic_decay(f, close, horizons=[1, 5, 21, 63])
    assert list(decay.index) == [1, 5, 21, 63]
    assert "rank_ic_mean" in decay.columns


def test_benjamini_hochberg_controls_fdr():
    # All-null p-values => nothing significant.
    pvals = pd.Series({f"f{i}": np.nan for i in range(5)})
    assert not benjamini_hochberg(pvals).any()
    # A mixture: tiny p (sig) and 1.0 (ns).
    p = pd.Series({"a": 0.001, "b": 0.5, "c": 0.9, "d": 0.001, "e": 0.8})
    passed = benjamini_hochberg(p, alpha=0.05)
    assert passed["a"] and passed["d"]
    assert not passed["b"] and not passed["c"]

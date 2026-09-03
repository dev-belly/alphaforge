"""Unit tests for the market-regime classification module.

The regime module must classify only from trailing information (no future
data), cover every valid label, and degrade gracefully on short series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.risk.regime import (
    REGIME_LABELS,
    classify_regime,
    factor_performance_by_regime,
    regime_statistics,
)


def _synth_returns(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    # Block-structured drift so a trend signal actually exists.
    drift = np.concatenate(
        [np.full(n // 3, 0.0006), np.full(n // 3, -0.0008), np.full(n - 2 * (n // 3), 0.0005)]
    )
    r = drift + rng.normal(0, 0.01, size=n)
    return pd.Series(r, index=idx)


def test_classify_covers_all_four_labels():
    r = _synth_returns(600, seed=3)
    labels = classify_regime(r)
    assert set(REGIME_LABELS).issubset(set(labels.dropna().unique()))


def test_classify_no_future_leakage():
    """The label for an early date must not depend on a late date's value."""
    r = _synth_returns(400, seed=7)
    full = classify_regime(r)
    # Truncating the history must not change already-computed early labels.
    early = classify_regime(r.iloc[:250])
    # The first decided date in `early` should match `full` at that index.
    decided = early.dropna()
    assert len(decided) > 0
    first_idx = decided.index[0]
    assert full.loc[first_idx] == early.loc[first_idx]


def test_classify_short_series_returns_empty():
    r = _synth_returns(50, seed=1)
    out = classify_regime(r)
    # Not enough history to decide -> empty (no labels, no crash).
    assert out.dropna().empty


def test_regime_statistics_sharpe_finite():
    r = _synth_returns(600, seed=5)
    regime = classify_regime(r)
    stats = regime_statistics(r, regime)
    assert stats  # at least one regime present
    for _lab, s in stats.items():
        assert s["n_days"] >= 5
        assert np.isfinite(s["ann_return"])
        assert s["ann_vol"] > 0
        assert np.isfinite(s["sharpe"]) or s["ann_vol"] <= 0


def test_factor_performance_by_regime_uses_ic():
    rng = np.random.default_rng(11)
    r = _synth_returns(600, seed=11)
    regime = classify_regime(r)
    ic = pd.Series(rng.normal(0.02, 0.05, size=len(r)), index=r.index)
    out = factor_performance_by_regime(ic, regime)
    assert out
    for _lab, s in out.items():
        assert s["n"] >= 5
        assert np.isfinite(s["ic_mean"])
        assert 0.0 <= s["positive_ic_ratio"] <= 1.0

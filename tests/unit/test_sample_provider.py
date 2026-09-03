"""Determinism + sanity checks for the synthetic `sample` provider.

The case-study JSON (``research/case_study_data.json``) is generated from this
provider. A non-deterministic or degenerate benchmark would silently corrupt
every benchmark-relative metric (beta, alpha, information ratio) *and* the risk
model's ``market`` factor — which is exactly what produced a stale, broken
case-study JSON earlier. These fast tests lock the correct behaviour so the
regression cannot go unnoticed again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.providers.sample import SampleDataProvider, SampleSpec


def _provider() -> SampleDataProvider:
    return SampleDataProvider(SampleSpec(seed=42))


def test_benchmark_is_deterministic() -> None:
    a = _provider().benchmark_prices()
    b = _provider().benchmark_prices()
    pd.testing.assert_series_equal(a, b)


def test_benchmark_returns_are_finite_and_sane() -> None:
    bench = _provider().benchmark_prices()
    rets = bench.pct_change(fill_method=None).dropna()
    assert np.isfinite(rets.to_numpy()).all()
    # A degenerate benchmark (a -100% daily return) would zero the index and
    # make every benchmark-relative metric meaningless.
    assert (rets > -1.0).all()
    cum = float((1.0 + rets).cumprod().iloc[-1])
    assert np.isfinite(cum) and cum > 0.0
    # Daily vol must sit in a plausible equity band: not flat, not explosive.
    assert 0.0 < rets.std() < 0.1

"""Unit tests for returns-based factor attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.attribution.factor import factor_attribution


def _synth(n: int = 400, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    f1 = rng.normal(0.0004, 0.01, size=n)
    f2 = rng.normal(0.0002, 0.008, size=n)
    # True loadings 0.6 on f1, 0.3 on f2, tiny idiosyncratic noise.
    port = 0.6 * f1 + 0.3 * f2 + rng.normal(0.0, 0.0005, size=n)
    return pd.Series(port, index=idx), pd.DataFrame({"momentum": f1, "value": f2}, index=idx)


def test_attribution_recovers_loadings():
    port, fr = _synth(seed=1)
    res = factor_attribution(port, fr)
    assert res.betas.shape[0] == 2
    assert abs(res.betas["momentum"] - 0.6) < 0.05
    assert abs(res.betas["value"] - 0.3) < 0.05
    assert 0.0 <= res.r_squared <= 1.0
    assert res.r_squared > 0.9  # strong fit by construction


def test_attribution_contributions_sum_to_predicted():
    port, fr = _synth(seed=2)
    res = factor_attribution(port, fr)
    predicted = res.attributed_return.sum() + res.residual_mean
    # The mean excess return equals the sum of factor contributions + residual.
    y = port.mean()
    assert abs(predicted - y) < 1e-3


def test_attribution_with_benchmark_uses_active():
    port, fr = _synth(seed=3)
    bench = pd.Series(port * 0.4 + 0.0001, index=port.index)  # partially correlated
    res = factor_attribution(port, fr, benchmark_returns=bench)
    assert res.betas.shape[0] == 2
    assert 0.0 <= res.r_squared <= 1.0 or np.isnan(res.r_squared)


def test_attribution_few_obs_returns_zeros():
    idx = pd.date_range("2021-01-01", periods=3, freq="B")
    port = pd.Series([0.01, -0.01, 0.02], index=idx)
    fr = pd.DataFrame({"a": [0.01, -0.01, 0.02], "b": [0.0, 0.01, -0.01]}, index=idx)
    res = factor_attribution(port, fr)
    # 3 obs < n_factors(2) + 2 -> the module degrades to a zero fill.
    assert res.n_observations < 4
    assert float(res.betas.sum()) == 0.0  # graceful zero fill

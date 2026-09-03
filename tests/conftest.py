"""Shared pytest fixtures for the AlphaForge test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def assets() -> list[str]:
    return [f"A{i:02d}" for i in range(20)]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=300, freq="B")


@pytest.fixture
def synthetic_returns(assets, dates, rng) -> pd.DataFrame:
    """Cross-sectionally-correlated daily returns, (dates x assets)."""
    n = len(assets)
    # A single market factor + idiosyncratic noise gives a realistic covariance.
    market = rng.normal(0.0003, 0.01, size=len(dates))
    beta = rng.uniform(0.4, 1.3, size=n)
    noise = rng.normal(0.0, 0.012, size=(len(dates), n))
    arr = market[:, None] * beta[None, :] + noise
    return pd.DataFrame(arr, index=dates, columns=assets)


@pytest.fixture
def synthetic_cov(synthetic_returns) -> pd.DataFrame:
    from alphaforge.risk.covariance import CovarianceEstimator

    est = CovarianceEstimator("ledoit_wolf").estimate(synthetic_returns)
    return est.matrix


@pytest.fixture
def synthetic_mu(assets, rng) -> pd.Series:
    return pd.Series(rng.uniform(-0.05, 0.10, size=len(assets)), index=assets)


@pytest.fixture
def synthetic_industry(assets) -> pd.Series:
    sectors = [
        "Tech",
        "Tech",
        "Fin",
        "Fin",
        "Fin",
        "Energy",
        "Energy",
        "Health",
        "Health",
        "Health",
        "Tech",
        "Fin",
        "Energy",
        "Health",
        "Tech",
        "Fin",
        "Energy",
        "Health",
        "Tech",
        "Fin",
    ]
    return pd.Series(sectors[: len(assets)], index=assets)

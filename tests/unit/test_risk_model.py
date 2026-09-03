"""Unit tests for the fundamental risk model and the Euler risk decomposition."""

from __future__ import annotations

import pandas as pd

from alphaforge.risk.factor_model import (
    FundamentalRiskModel,
    RiskModelConfig,
    component_risk_contribution,
    factor_risk_decomposition,
    portfolio_risk,
)


def _fit(synthetic_returns, synthetic_industry):
    mc = pd.Series(1e9, index=synthetic_returns.columns)
    style = {
        "value": synthetic_returns.mean(axis=0),  # any cross-sectional proxy works
    }
    rm = FundamentalRiskModel(RiskModelConfig(style_factors=["size", "value"]))
    return rm.fit(synthetic_returns, mc, synthetic_industry, style)


def test_fit_returns_decomposition_components(synthetic_returns, synthetic_industry):
    res = _fit(synthetic_returns, synthetic_industry)
    assert res.exposures.shape[0] == synthetic_returns.shape[1]
    assert res.factor_cov.shape[0] == res.exposures.shape[1]
    # covariance is the asset-level (n x n) matrix, exposures are (n x factors).
    assert res.covariance.shape[0] == res.exposures.shape[0]
    assert res.covariance.shape[0] == res.covariance.shape[1]
    assert 0.0 <= res.r_squared <= 1.0
    # factor_returns must align to the returns index
    assert res.factor_returns.index.equals(synthetic_returns.index)


def test_euler_decomposition_holds(synthetic_returns, synthetic_industry):
    """Sum of component risk contributions equals portfolio volatility (Euler)."""
    res = _fit(synthetic_returns, synthetic_industry)
    w = pd.Series(1.0 / len(res.covariance.columns), index=res.covariance.columns)
    port_vol = portfolio_risk(w, res.covariance)
    crc = component_risk_contribution(w, res.covariance)
    assert abs(crc.sum() - port_vol) < 1e-6, "Euler identity violated"
    assert (crc.to_numpy() >= -1e-9).all()


def test_factor_risk_decomposition_sums_to_variance(synthetic_returns, synthetic_industry):
    res = _fit(synthetic_returns, synthetic_industry)
    w = pd.Series(1.0 / len(res.exposures.index), index=res.exposures.index)
    dec = factor_risk_decomposition(w, res.exposures, res.factor_cov, res.specific_var)

    B = res.exposures.to_numpy(dtype=float)
    F = res.factor_cov.reindex(index=res.exposures.columns, columns=res.exposures.columns).to_numpy(
        dtype=float
    )
    spec = res.specific_var.reindex(res.exposures.index).to_numpy(dtype=float)
    b_p = w.to_numpy() @ B
    total_var = float(b_p @ F @ b_p) + float((w.to_numpy() ** 2 * spec).sum())

    pct_sum = dec["contribution_to_variance"].sum()
    assert abs(pct_sum - total_var) < 1e-6

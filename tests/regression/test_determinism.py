"""Regression tests: the engine must be deterministic for a fixed seed.

These are cheap (synthetic inputs) but lock the numerical behaviour so a
refactor that silently changes solver/seed handling is caught immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
from alphaforge.risk.factor_model import FundamentalRiskModel, RiskModelConfig


def test_optimizer_is_deterministic(synthetic_mu, synthetic_cov, synthetic_industry):
    cfg = OptimizerConfig(method="mean_variance")
    a = (
        PortfolioOptimizer(cfg)
        .solve(synthetic_mu, synthetic_cov, industry=synthetic_industry)
        .weights
    )
    b = (
        PortfolioOptimizer(cfg)
        .solve(synthetic_mu, synthetic_cov, industry=synthetic_industry)
        .weights
    )
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-10)


def test_risk_model_fit_is_deterministic(synthetic_returns, synthetic_industry):
    mc = pd.Series(1e9, index=synthetic_returns.columns)
    rm = FundamentalRiskModel(RiskModelConfig(style_factors=["size", "value"]))
    r1 = rm.fit(
        synthetic_returns, mc, synthetic_industry, {"value": synthetic_returns.mean(axis=0)}
    )
    r2 = rm.fit(
        synthetic_returns, mc, synthetic_industry, {"value": synthetic_returns.mean(axis=0)}
    )
    assert np.allclose(r1.exposures.to_numpy(), r2.exposures.to_numpy(), atol=1e-10)
    assert np.allclose(r1.factor_cov.to_numpy(), r2.factor_cov.to_numpy(), atol=1e-10)

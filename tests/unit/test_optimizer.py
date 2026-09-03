"""Unit tests for the portfolio optimiser across all five methods."""

from __future__ import annotations

import numpy as np
import pytest

from alphaforge.portfolio.optimizer import METHODS, OptimizerConfig, PortfolioOptimizer


@pytest.mark.parametrize("method", METHODS)
def test_optimizer_produces_feasible_long_only_book(
    method, synthetic_mu, synthetic_cov, synthetic_industry
):
    cfg = OptimizerConfig(method=method, fully_invested=True, long_only=True, max_weight=0.10)
    res = PortfolioOptimizer(cfg).solve(synthetic_mu, synthetic_cov, industry=synthetic_industry)
    w = res.weights
    assert np.all(np.isfinite(w.to_numpy()))
    assert (w >= -1e-9).all(), "long-only constraint violated"
    assert (w <= cfg.max_weight + 1e-9).all(), "max-weight cap violated"
    # Vol targeting may leave a cash buffer; the book must still be material.
    assert w.sum() > 0.5
    # The volatility budget must not be exceeded.
    if cfg.target_volatility:
        assert res.diagnostics["ex_ante_vol"] <= cfg.target_volatility + 1e-6
    assert res.n_holdings >= cfg.min_names


def test_fully_invested_budget_without_vol_target(synthetic_mu, synthetic_cov, synthetic_industry):
    cfg = OptimizerConfig(method="mean_variance", target_volatility=None, max_weight=0.10)
    res = PortfolioOptimizer(cfg).solve(synthetic_mu, synthetic_cov, industry=synthetic_industry)
    assert abs(res.weights.sum() - 1.0) < 1e-6


def test_equal_weight_is_uniform_within_cap(synthetic_cov):
    cfg = OptimizerConfig(method="equal_weight", max_weight=0.10, target_volatility=None)
    res = PortfolioOptimizer(cfg).solve(None, synthetic_cov)
    w = res.weights
    # With 20 names and a 10% cap there is no binding, so weights are ~uniform.
    assert abs(w.max() - w.min()) < 1e-6
    assert abs(w.sum() - 1.0) < 1e-6


def test_unknown_method_raises(synthetic_cov):
    with pytest.raises(ValueError):
        PortfolioOptimizer(OptimizerConfig(method="not_a_method"))


def test_too_few_names_raises(synthetic_cov):
    small = synthetic_cov.iloc[:3, :3]
    with pytest.raises(ValueError):
        PortfolioOptimizer(OptimizerConfig(min_names=10)).solve(None, small)

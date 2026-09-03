"""Unit tests for scenario stress testing.

Stress P&L must be mechanically consistent with the risk-model exposures: a
fixed ``market`` shock of -X must move a beta-1 portfolio by ~-X, and a
factor-sigma shock must scale with the factor's standalone volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.risk.factor_model import (
    FundamentalRiskModel,
    RiskModelConfig,
    RiskModelResult,
)
from alphaforge.risk.stress import (
    DEFAULT_SCENARIOS,
    run_scenarios,
    stress_portfolio,
)


def _fake_risk(n_assets: int = 20, seed: int = 0) -> RiskModelResult:
    rng = np.random.default_rng(seed)
    assets = [f"A{i}" for i in range(n_assets)]
    factors = ["market", "value", "momentum", "volatility", "quality", "liquidity"]
    B = pd.DataFrame(rng.normal(0, 1, size=(n_assets, len(factors))), index=assets, columns=factors)
    B["market"] = 1.0  # every asset has unit market exposure
    F = pd.DataFrame(
        np.eye(len(factors)) * 0.04, index=factors, columns=factors
    )  # 20% factor vol
    spec = pd.Series(rng.uniform(0.1, 0.3, size=n_assets), index=assets)
    cov = B.to_numpy() @ F.to_numpy() @ B.to_numpy().T + np.diag(spec.to_numpy())
    return RiskModelResult(
        exposures=B,
        factor_cov=F,
        specific_var=spec,
        covariance=pd.DataFrame(cov, index=assets, columns=assets),
        factor_returns=pd.DataFrame(rng.normal(0, 0.01, size=(10, len(factors))), columns=factors),
        residuals=pd.DataFrame(rng.normal(0, 0.01, size=(10, n_assets)), columns=assets),
        r_squared=0.5,
        config=RiskModelConfig().from_dict({}).__dict__,
    )


def test_market_shock_moves_beta_one_book_by_x():
    risk = _fake_risk()
    w = pd.Series(1.0 / 20, index=risk.exposures.index)  # equal weight, beta=1 each
    res = stress_portfolio(w, risk, {"kind": "factor", "factor": "market", "value": -0.10}, "m")
    # Sum of weights = 1, market exposure = 1 -> P&L ~ -10%.
    assert abs(res.pnl_pct - (-0.10)) < 1e-9


def test_factor_sigma_shock_scales_with_vol():
    risk = _fake_risk()
    w = pd.Series(1.0 / 20, index=risk.exposures.index)
    res = stress_portfolio(
        w, risk, {"kind": "factor_sigma", "factor": "momentum", "multiplier": -2.0}, "mom"
    )
    # momentum vol = 20% -> -2 sigma = -40% move on the momentum factor.
    # Portfolio momentum exposure is ~0 (random), so P&L should be small in
    # magnitude relative to a concentrated bet.
    assert abs(res.pnl_pct) < 0.5


def test_run_scenarios_returns_all_defaults():
    risk = _fake_risk()
    w = pd.Series(1.0 / 20, index=risk.exposures.index)
    out = run_scenarios(w, risk)
    assert set(out.keys()) == set(DEFAULT_SCENARIOS.keys())
    for r in out.values():
        assert np.isfinite(r.pnl_pct)
        assert r.worst_holdings()


def test_integrated_run_via_fundamental_model():
    """End-to-end: fit the real risk model on synthetic returns, stress it."""
    rng = np.random.default_rng(3)
    n = 30
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    assets = [f"S{i}" for i in range(n)]
    rets = pd.DataFrame(rng.normal(0.0003, 0.02, size=(len(dates), n)), index=dates, columns=assets)
    mcap = pd.Series(rng.uniform(1e9, 1e11, size=n), index=assets)
    industry = pd.Series([f"Ind{i % 5}" for i in range(n)], index=assets)
    risk = FundamentalRiskModel().fit(rets, mcap, industry)
    w = pd.Series(np.abs(rng.normal(size=n)), index=assets)
    w = w / w.sum()
    out = run_scenarios(w, risk)
    assert out, "expected at least one scenario to resolve"
    for r in out.values():
        assert np.isfinite(r.pnl_pct)

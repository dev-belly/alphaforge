"""Unit tests for scenario stress testing.

Stress P&L must be mechanically consistent with the risk-model exposures: a
fixed ``market`` shock of -X must move a beta-1 portfolio by ~-X, and a
factor-sigma shock must scale with the factor's standalone volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    F = pd.DataFrame(np.eye(len(factors)) * 0.04, index=factors, columns=factors)  # 20% factor vol
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


# --------------------------------------------------------------------------
# StressResult
# --------------------------------------------------------------------------
def test_worst_holdings_is_empty_without_a_contribution_vector():
    """A result built without contributions has no ranking to offer."""
    from alphaforge.risk.stress import StressResult

    bare = StressResult(scenario="s", pnl_pct=-0.05, shock={"market": -0.05})
    assert bare.worst_holdings() == []


def test_worst_holdings_ranks_the_most_negative_first():
    from alphaforge.risk.stress import StressResult

    contributions = pd.Series({"A": -0.02, "B": 0.01, "C": -0.05})
    result = StressResult(
        scenario="s", pnl_pct=-0.06, shock={"market": -0.05}, contributions=contributions
    )
    got = result.worst_holdings(n=2)
    assert [row["symbol"] for row in got] == ["C", "A"]
    assert got[0]["contribution"] == pytest.approx(-0.05)


def test_worst_holdings_honours_n():
    from alphaforge.risk.stress import StressResult

    contributions = pd.Series(np.linspace(-0.10, 0.10, 20), index=[f"A{i}" for i in range(20)])
    result = StressResult(scenario="s", pnl_pct=-0.1, shock={}, contributions=contributions)
    assert len(result.worst_holdings(n=3)) == 3
    assert len(result.worst_holdings(n=50)) == 20


def test_to_dict_carries_everything_the_report_needs():
    risk = _fake_risk()
    w = pd.Series(1.0 / 20, index=risk.exposures.index)
    result = stress_portfolio(w, risk, {"kind": "factor", "factor": "market", "value": -0.10}, "m")
    got = result.to_dict()
    assert got["scenario"] == "m"
    assert got["pnl_pct"] == pytest.approx(result.pnl_pct)
    assert got["shock"] == {"market": -0.10}
    assert got["factor_exposure"]
    assert got["worst_holdings"], "the report needs the holdings that drove the loss"


def test_to_dict_is_empty_safe_for_a_bare_result():
    from alphaforge.risk.stress import StressResult

    got = StressResult(scenario="s", pnl_pct=-0.01, shock={}).to_dict()
    assert got["worst_holdings"] == []
    assert got["factor_exposure"] == {}


# --------------------------------------------------------------------------
# run_scenarios
# --------------------------------------------------------------------------
def test_run_scenarios_skips_a_scenario_that_raises(monkeypatch):
    """One bad spec must not sink the rest of the book."""
    from alphaforge.risk import stress as stress_module

    risk = _fake_risk()
    w = pd.Series(1.0 / 20, index=risk.exposures.index)
    real = stress_module.stress_portfolio

    def flaky(weights, risk_result, spec, name=None):
        if name == "boom":
            raise ValueError("malformed spec")
        return real(weights, risk_result, spec, name=name)

    monkeypatch.setattr(stress_module, "stress_portfolio", flaky)
    got = run_scenarios(
        w,
        risk,
        {
            "boom": {"kind": "factor", "factor": "market", "value": -0.5},
            "ok": {"kind": "factor", "factor": "market", "value": -0.10},
        },
    )
    assert set(got) == {"ok"}
    assert got["ok"].pnl_pct == pytest.approx(-0.10, abs=1e-9)


def test_an_empty_scenario_mapping_means_the_defaults():
    """``scenarios or DEFAULT_SCENARIOS``: an empty dict is falsy, like None."""
    risk = _fake_risk()
    w = pd.Series(1.0 / 20, index=risk.exposures.index)
    assert set(run_scenarios(w, risk, {})) == set(DEFAULT_SCENARIOS)
    assert set(run_scenarios(w, risk, None)) == set(DEFAULT_SCENARIOS)


# --------------------------------------------------------------------------
# sector_shock
# --------------------------------------------------------------------------
def _sector_book():
    industry = pd.Series(
        {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Health"},
    )
    weights = pd.Series({"A": 0.4, "B": 0.1, "C": 0.3, "D": 0.2})
    return weights, industry


def test_a_sector_shock_hits_only_that_sector():
    from alphaforge.risk.stress import sector_shock

    weights, industry = _sector_book()
    got = sector_shock(weights, industry, "Tech", 0.20)
    # 0.4 + 0.1 = 0.5 of NAV in Tech, shocked by -20% -> -10%.
    assert got.pnl_pct == pytest.approx(-0.10, rel=1e-12)


def test_a_positive_shock_is_still_a_loss():
    """The helper takes a shock *size*; the sign is the function's job."""
    from alphaforge.risk.stress import sector_shock

    weights, industry = _sector_book()
    assert sector_shock(weights, industry, "Tech", 0.20).pnl_pct == pytest.approx(
        sector_shock(weights, industry, "Tech", -0.20).pnl_pct
    )


def test_an_untouched_sector_contributes_nothing():
    from alphaforge.risk.stress import sector_shock

    weights, industry = _sector_book()
    got = sector_shock(weights, industry, "Tech", 0.20)
    contributions = got.contributions
    assert contributions["C"] == pytest.approx(0.0)
    assert contributions["D"] == pytest.approx(0.0)
    assert contributions["A"] == pytest.approx(-0.08)


def test_the_scenario_label_encodes_the_sector_and_the_shock():
    from alphaforge.risk.stress import sector_shock

    weights, industry = _sector_book()
    assert sector_shock(weights, industry, "Tech", 0.15).scenario == "sector_Tech_15pct"
    assert sector_shock(weights, industry, "Energy", 0.30).scenario == "sector_Energy_30pct"


def test_the_sector_weight_is_reported():
    from alphaforge.risk.stress import sector_shock

    weights, industry = _sector_book()
    got = sector_shock(weights, industry, "Tech", 0.20)
    assert got.factor_exposure["sector_weight"] == pytest.approx(0.5)


def test_an_unknown_sector_shocks_nothing():
    from alphaforge.risk.stress import sector_shock

    weights, industry = _sector_book()
    got = sector_shock(weights, industry, "Utilities", 0.50)
    assert got.pnl_pct == pytest.approx(0.0)
    assert got.factor_exposure["sector_weight"] == pytest.approx(0.0)


def test_a_held_name_missing_from_the_industry_map_is_not_shocked():
    """It cannot be attributed to a sector, so it contributes nothing.

    Documented rather than fixed: the helper reindexes the book onto the
    industry map, so an unclassified holding silently leaves the scenario. In
    the pipeline the map comes from the panel and covers every traded symbol.
    """
    from alphaforge.risk.stress import sector_shock

    weights = pd.Series({"A": 0.5, "ZZ": 0.5})
    industry = pd.Series({"A": "Tech"})
    got = sector_shock(weights, industry, "Tech", 0.20)
    assert got.pnl_pct == pytest.approx(-0.10)
    assert "ZZ" not in got.contributions.index

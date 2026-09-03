"""Scenario stress testing for the factor-exposed portfolio.

Stress testing answers "*what happens to the portfolio if factor X moves by a
given amount?*" It is complementary to the covariance-based risk model: the
risk model tells you the **distribution** of outcomes, stress testing pins a
**single adverse path** and reports the P&L.

Every scenario is expressed as a shock to one or more **risk-model factors**
(the same factors the portfolio is already exposed to), so the result is
internally consistent with the risk decomposition::

    portfolio_factor_exposure = weights' @ B          # (n_factors,)
    scenario_pnl             = portfolio_factor_exposure @ shock_vector

Two shock kinds are supported:

* ``factor``      - a fixed move, e.g. ``market -> -10%``.
* ``factor_sigma``- a move of ``k * sigma`` where ``sigma`` is the factor's
  standalone volatility from the factor covariance (e.g. ``momentum -> -2σ``).

This is deterministic and fully reproducible (see Master Prompt, "Stress
Testing"). It never invents a number: the shocks are explicit inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.risk.factor_model import RiskModelResult

# Default scenario book. Each entry is an explicit, auditable shock.
DEFAULT_SCENARIOS: dict[str, dict] = {
    "market_drawdown_10pct": {"kind": "factor", "factor": "market", "value": -0.10},
    "momentum_crash_2sigma": {"kind": "factor_sigma", "factor": "momentum", "multiplier": -2.0},
    "value_selloff_5pct": {"kind": "factor", "factor": "value", "value": -0.05},
    "volatility_spike_3sigma": {"kind": "factor_sigma", "factor": "volatility", "multiplier": 3.0},
    "quality_rotation_5pct": {"kind": "factor", "factor": "quality", "value": 0.05},
    "liquidity_dryup_2sigma": {"kind": "factor_sigma", "factor": "liquidity", "multiplier": -2.0},
}


@dataclass
class StressResult:
    """Outcome of one stress scenario applied to the portfolio."""

    scenario: str
    pnl_pct: float
    shock: dict[str, float]
    factor_exposure: dict[str, float] = field(default_factory=dict)
    contributions: pd.Series | None = None

    def worst_holdings(self, n: int = 5) -> list[dict]:
        if self.contributions is None:
            return []
        c = self.contributions.sort_values()
        return [
            {"symbol": str(idx), "contribution": float(v)} for idx, v in c.head(n).items()
        ]

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "pnl_pct": self.pnl_pct,
            "shock": self.shock,
            "factor_exposure": self.factor_exposure,
            "worst_holdings": self.worst_holdings(),
        }


def _build_shock_vector(scenario: dict, risk: RiskModelResult) -> pd.Series:
    """Resolve a scenario spec into a per-factor shock vector."""
    cols = list(risk.exposures.columns)
    shock = pd.Series(0.0, index=cols)
    factor = scenario.get("factor")
    if factor is None or factor not in cols:
        return shock
    if scenario.get("kind") == "factor_sigma":
        mult = float(scenario.get("multiplier", -2.0))
        var = risk.factor_cov.loc[factor, factor] if factor in risk.factor_cov.index else np.nan
        sigma = float(np.sqrt(max(var, 0.0)))
        shock[factor] = mult * sigma
    else:  # fixed "factor" shock
        shock[factor] = float(scenario.get("value", 0.0))
    return shock


def stress_portfolio(
    weights: pd.Series,
    risk: RiskModelResult,
    scenario: dict,
    name: str = "scenario",
) -> StressResult:
    """Apply one scenario (spec dict) to ``weights`` under ``risk``."""
    w = weights.reindex(risk.exposures.index).fillna(0.0)
    port_exp = w.to_numpy(dtype=float) @ risk.exposures.to_numpy(dtype=float)  # (n_factors,)
    shock = _build_shock_vector(scenario, risk)
    pnl = float(port_exp @ shock.to_numpy(dtype=float))

    asset_shock = risk.exposures.to_numpy(dtype=float) @ shock.to_numpy(dtype=float)
    contributions = pd.Series(w.to_numpy(dtype=float) * asset_shock, index=risk.exposures.index)

    return StressResult(
        scenario=name,
        pnl_pct=pnl,
        shock={k: float(v) for k, v in shock.items() if v != 0.0},
        factor_exposure={
            c: float(port_exp[i]) for i, c in enumerate(risk.exposures.columns)
        },
        contributions=contributions,
    )


def run_scenarios(
    weights: pd.Series,
    risk: RiskModelResult,
    scenarios: dict[str, dict] | None = None,
) -> dict[str, StressResult]:
    """Run a book of scenarios; defaults to :data:`DEFAULT_SCENARIOS`."""
    scenarios = scenarios or DEFAULT_SCENARIOS
    out: dict[str, StressResult] = {}
    for nm, spec in scenarios.items():
        try:
            out[nm] = stress_portfolio(weights, risk, spec, name=nm)
        except Exception:  # noqa: BLE001 - one bad scenario must not sink the rest
            continue
    return out


# --------------------------------------------------------------------------
# sector shock helper (uses an industry map, not a risk factor)
# --------------------------------------------------------------------------
def sector_shock(
    weights: pd.Series,
    industry: pd.Series,
    sector: str,
    shock_pct: float,
) -> StressResult:
    """Direct, asset-level shock to one GICS sector (no factor model needed)."""
    w = weights.reindex(industry.index).fillna(0.0)
    in_sector = (industry.astype(str) == sector).reindex(w.index).fillna(False).astype(float)
    asset_shock = -abs(shock_pct) * in_sector
    contributions = w * asset_shock
    return StressResult(
        scenario=f"sector_{sector}_{int(shock_pct * 100)}pct",
        pnl_pct=float(contributions.sum()),
        shock={sector: float(shock_pct)},
        factor_exposure={"sector_weight": float(w[in_sector.astype(bool)].sum())},
        contributions=contributions,
    )


__all__ = [
    "DEFAULT_SCENARIOS",
    "StressResult",
    "stress_portfolio",
    "run_scenarios",
    "sector_shock",
]

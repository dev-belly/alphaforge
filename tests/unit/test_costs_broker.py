"""Unit tests for the execution layer (cost model + broker simulation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.execution.broker import BrokerSimulator
from alphaforge.execution.costs import CostModel, total_cost_bps


def test_cost_model_zero_for_zero_value():
    cm = CostModel({"commission_bps": 5, "slippage_bps": 10, "impact_coeff_bps": 20})
    c = cm.estimate(0.0, 1e7)
    assert c.total == 0.0


def test_impact_scales_with_sqrt_participation():
    cm = CostModel({"impact_coeff_bps": 100.0, "participation_cap": 0.5})
    # participation 0.04 -> impact = 100*sqrt(0.04)=20bps; 0.25 -> 50bps.
    lo = cm.impact_bps(0.04)
    hi = cm.impact_bps(0.25)
    assert abs(lo - 20.0) < 1e-6
    assert abs(hi - 50.0) < 1e-6
    assert hi > lo  # sub-linear but strictly increasing


def test_estimate_has_three_components():
    cm = CostModel({"commission_bps": 5, "slippage_bps": 10, "impact_coeff_bps": 20})
    c = cm.estimate(2_000_000.0, 1_000_000.0)
    assert c.commission > 0 and c.slippage > 0 and c.impact > 0
    assert c.total == c.commission + c.slippage + c.impact


def test_participation_capped():
    cm = CostModel({"participation_cap": 0.10})
    # A huge order is capped at the participation cap, not exceeded.
    assert cm.participation(1e12, 1e6) == 0.10


def test_total_cost_bps():
    costs = pd.DataFrame({"cost_total": [100.0, 200.0]})
    assert total_cost_bps(costs, 1_000.0) == 3000.0  # 3000 bps of notional


def _rebalance_setup(n: int = 5):
    syms = [f"S{i}" for i in range(n)]
    target = pd.Series(1.0 / n, index=syms)
    current = pd.Series(0.0, index=syms)
    prices = pd.Series(100.0, index=syms)
    adv = pd.Series(1e7, index=syms)
    return syms, target, current, prices, adv


def test_broker_charges_cost_on_trade():
    syms, target, current, prices, adv = _rebalance_setup()
    res = BrokerSimulator(
        {"commission_bps": 5, "slippage_bps": 10, "impact_coeff_bps": 20}
    ).rebalance(target, current, nav=1e8, prices=prices, adv_value=adv)
    assert not res.trades.empty
    assert res.cost_total > 0
    assert res.cost_bps > 0
    # Realised weights re-expressed on post-cost NAV (gross exposure > 1).
    assert res.weights.reindex(syms).sum() > 1.0


def test_broker_untradeable_name_stays():
    syms, target, current, prices, adv = _rebalance_setup()
    # One name has no price and a non-zero current weight -> must stay, not sell.
    prices.loc["S0"] = np.nan
    current.loc["S0"] = 0.3
    target.loc["S0"] = 0.5  # want to change it, but cannot trade
    res = BrokerSimulator(
        {"commission_bps": 5, "slippage_bps": 10, "impact_coeff_bps": 20}
    ).rebalance(target, current, nav=1e8, prices=prices, adv_value=adv)
    assert "S0" in res.unfillable
    # The stuck name keeps its pre-trade weight.
    assert abs(res.weights.loc["S0"] - 0.3) < 1e-9


def test_broker_no_trade_when_already_there():
    syms, target, current, prices, adv = _rebalance_setup()
    current = target.copy()
    res = BrokerSimulator().rebalance(target, current, nav=1e8, prices=prices, adv_value=adv)
    assert res.trades.empty
    assert res.cost_total == 0.0

"""Unit tests for the execution layer (cost model + broker simulation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.execution.broker import BrokerConfig, BrokerSimulator
from alphaforge.execution.costs import CostConfig, CostModel, total_cost_bps


@pytest.mark.parametrize(
    "settings",
    [
        {"slippage_bps": -50},
        {"commission_bps": -1},
        {"impact_coeff_bps": float("nan")},
        {"min_commission": float("inf")},
        {"borrow_bps_annual": -1},
        {"participation_cap": 0},
        {"participation_cap": 1.1},
        {"slippage_bps": True},
    ],
)
def test_unsafe_cost_assumptions_cannot_create_credits(settings: dict) -> None:
    with pytest.raises(ValueError):
        CostModel(settings)
    with pytest.raises(ValueError):
        CostConfig(**settings)


def test_string_false_enforces_integer_fills() -> None:
    broker = BrokerSimulator(
        {"commission_bps": 0, "slippage_bps": 0, "impact_coeff_bps": 0},
        {"allow_fractional_shares": "false"},
    )
    result = broker.rebalance(pd.Series({"A": 0.9}), None, 100.0, pd.Series({"A": 60.0}))
    assert result.trades.loc["A", "shares"] == 1
    assert result.weights.loc["A"] == pytest.approx(0.6)
    assert BrokerConfig.from_dict({"allow_fractional_shares": "true"}).allow_fractional_shares


@pytest.mark.parametrize(
    "settings",
    [
        {"allow_fractional_shares": "maybe"},
        {"allow_fractional_shares": 1},
        {"participation_cap": 0},
        {"min_trade_value": -1},
        {"min_trade_value": float("nan")},
    ],
)
def test_unsafe_broker_controls_fail_before_any_fill(settings: dict) -> None:
    with pytest.raises(ValueError):
        BrokerSimulator(config=settings)


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
    # Its shares stay put; its weight rises slightly when fees reduce NAV.
    assert res.weights.loc["S0"] == pytest.approx(0.3 * 1e8 / (1e8 - res.cost_total))


def test_broker_no_trade_when_already_there():
    syms, target, current, prices, adv = _rebalance_setup()
    current = target.copy()
    res = BrokerSimulator().rebalance(target, current, nav=1e8, prices=prices, adv_value=adv)
    assert res.trades.empty
    assert res.cost_total == 0.0


def test_broker_preserves_intended_cash_weight():
    broker = BrokerSimulator({"commission_bps": 0, "slippage_bps": 0, "impact_coeff_bps": 0})
    res = broker.rebalance(
        target_weights=pd.Series({"A": 0.5}),
        current_weights=pd.Series({"A": 0.0}),
        nav=100.0,
        prices=pd.Series({"A": 10.0}),
    )
    assert res.weights.loc["A"] == pytest.approx(0.5)
    assert res.trades.loc["A", "trade_value"] == pytest.approx(50.0)


def test_broker_does_not_buy_zero_price_or_deploy_reserved_cash():
    broker = BrokerSimulator({"commission_bps": 0, "slippage_bps": 0, "impact_coeff_bps": 0})
    res = broker.rebalance(
        target_weights=pd.Series({"A": 0.2, "B": 0.2}),
        current_weights=pd.Series({"A": 0.0, "B": 0.0}),
        nav=100.0,
        prices=pd.Series({"A": 10.0, "B": 0.0}),
    )
    assert res.weights.loc["A"] == pytest.approx(0.2)
    assert res.weights.loc["B"] == pytest.approx(0.0)
    assert res.unfillable == ["B"]


def test_integer_share_rounding_changes_actual_book_and_costs():
    broker = BrokerSimulator(
        {"commission_bps": 100, "slippage_bps": 0, "impact_coeff_bps": 0},
        {"allow_fractional_shares": False},
    )
    res = broker.rebalance(
        target_weights=pd.Series({"A": 0.9}),
        current_weights=pd.Series({"A": 0.0}),
        nav=100.0,
        prices=pd.Series({"A": 60.0}),
    )
    assert res.trades.loc["A", "shares"] == 1
    assert res.trades.loc["A", "trade_value"] == pytest.approx(60.0)
    assert res.cost_total == pytest.approx(0.6)
    assert res.weights.loc["A"] == pytest.approx(60.0 / 99.4)

    too_small = broker.rebalance(
        target_weights=pd.Series({"A": 0.5}),
        current_weights=pd.Series({"A": 0.0}),
        nav=100.0,
        prices=pd.Series({"A": 60.0}),
    )
    assert too_small.trades.empty
    assert too_small.cost_total == 0.0
    assert too_small.weights.loc["A"] == 0.0

    rounded_below_minimum = BrokerSimulator(
        {"commission_bps": 100, "slippage_bps": 0, "impact_coeff_bps": 0},
        {"allow_fractional_shares": False, "min_trade_value": 80.0},
    ).rebalance(
        target_weights=pd.Series({"A": 0.9}),
        current_weights=pd.Series({"A": 0.0}),
        nav=100.0,
        prices=pd.Series({"A": 60.0}),
    )
    assert rounded_below_minimum.trades.empty

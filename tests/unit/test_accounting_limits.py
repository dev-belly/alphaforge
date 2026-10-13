"""Numerical boundaries must fail clearly without corrupting accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtest.engine import BacktestEngine
from alphaforge.execution.broker import BrokerSimulator
from alphaforge.execution.costs import CostModel, TradeCost
from alphaforge.features.panel import MarketPanel
from alphaforge.utils.math_utils import compound


def _costs(**kwargs):
    return {"commission_bps": 0, "slippage_bps": 0, "impact_coeff_bps": 0, **kwargs}


@pytest.mark.parametrize("component", ["commission_bps", "slippage_bps", "impact_coeff_bps"])
def test_cost_scaling_keeps_a_representable_large_result(component):
    model = CostModel(_costs(**{component: 10_000}, participation_cap=1))
    assert model.estimate(1e308, 1e308).total == pytest.approx(1e308)


@pytest.mark.parametrize(
    "settings",
    [
        {"commission_bps": 20_000},
        {"slippage_bps": 20_000},
        {"impact_coeff_bps": 20_000, "participation_cap": 1},
        {"commission_bps": 10_000, "slippage_bps": 10_000},
    ],
)
def test_costs_outside_float_range_are_rejected(settings):
    with pytest.raises(ValueError, match="finite"):
        CostModel(_costs(**settings)).estimate(1e308, 1e308)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_nonfinite_trade_values_are_rejected(value):
    with pytest.raises(ValueError, match="trade_value"):
        CostModel().estimate(value, 1e6)


def test_broker_rejects_fees_larger_than_available_nav():
    broker = BrokerSimulator(_costs(commission_bps=20_000))
    with pytest.raises(ValueError, match="cost.*NAV"):
        broker.rebalance(pd.Series({"A": 1.0}), None, 100.0, pd.Series({"A": 10.0}))


def test_broker_allows_fees_that_exactly_exhaust_nav():
    result = BrokerSimulator(_costs(commission_bps=10_000)).rebalance(
        pd.Series({"A": 1.0}), None, 100.0, pd.Series({"A": 10.0})
    )
    assert result.cost_total == 100.0
    assert np.isfinite(result.weights).all()


def test_broker_keeps_representable_short_weights_without_intermediate_overflow():
    result = BrokerSimulator(_costs()).rebalance(
        pd.Series({"A": -1.9}),
        pd.Series({"A": -2.0}),
        1e308,
        pd.Series({"A": 1.0}),
        pd.Series({"A": 1e308}),
    )
    assert result.weights.loc["A"] == pytest.approx(-1.9)


def test_broker_rejects_unrepresentable_traded_share_quantity():
    with pytest.raises(ValueError, match="Traded shares.*representable"):
        BrokerSimulator(_costs()).rebalance(
            pd.Series({"A": 1.0}),
            None,
            1.0,
            pd.Series({"A": np.nextafter(0.0, 1.0)}),
        )


@pytest.mark.parametrize("nav", [np.inf, -np.inf, np.nan, -1.0])
def test_broker_rejects_invalid_nav(nav):
    with pytest.raises(ValueError, match="NAV"):
        BrokerSimulator(_costs()).rebalance(
            pd.Series({"A": 1.0}), None, nav, pd.Series({"A": 10.0})
        )


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_compound_rejects_nonfinite_returns_instead_of_a_fake_wipeout(value):
    with pytest.raises(ValueError, match="returns"):
        compound(pd.Series([0.1, value]))


def test_compound_rejects_losses_below_total_capital():
    with pytest.raises(ValueError, match="returns"):
        compound(pd.Series([0.1, -1.1]))


def test_compound_rejects_unrepresentable_wealth():
    with pytest.raises(ValueError, match="wealth"):
        compound(pd.Series([1e200, 1e200]))


def test_compound_rejects_underflow_instead_of_a_zero_curve_that_resurrects():
    with pytest.raises(ValueError, match="wealth.*representable"):
        compound(pd.Series([-0.999999] * 60 + [1e308, 1e308]))


def test_compound_keeps_zero_wealth_after_a_legal_total_loss():
    assert compound(pd.Series([0.1, -1.0, 0.1])).tolist() == pytest.approx([1.1, 0.0, 0.0])


def _panel(close_values):
    dates = pd.bdate_range("2024-01-01", periods=len(close_values))
    close = pd.DataFrame({"A": close_values}, index=dates)
    volume = pd.DataFrame(1e6, index=dates, columns=["A"])
    return MarketPanel(
        dates=dates,
        close=close,
        raw_close=close.copy(),
        returns=close.pct_change(fill_method=None),
        volume=volume,
        dollar_volume=close * volume,
        market_cap=pd.DataFrame(1e9, index=dates, columns=["A"]),
        universe=close.notna(),
        industry=pd.DataFrame("Other", index=dates, columns=["A"]),
    )


def _engine(panel, *, target=1.0, initial_capital=100.0, cost_model=None, allow_short=False):
    return BacktestEngine(
        panel,
        weight_fn=lambda _date, _weights: pd.Series({"A": target}),
        cost_model=_costs() if cost_model is None else cost_model,
        config={
            "rebalance": "daily",
            "min_history_days": 0,
            "initial_capital": initial_capital,
            "allow_short": allow_short,
        },
    )


def test_engine_rejects_an_extreme_finite_fee_with_a_clear_value_error():
    with pytest.raises(ValueError, match="cost.*NAV"):
        _engine(_panel([10.0] * 8), cost_model=_costs(commission_bps=1e308)).run()


@pytest.mark.parametrize("fee", [-1.0, np.inf, 101.0])
def test_custom_broker_cannot_bypass_the_engine_fee_boundary(fee):
    class FixedFeeBroker(BrokerSimulator):
        def rebalance(self, **kwargs):
            result = super().rebalance(**kwargs)
            result.cost_total = fee
            return result

    engine = _engine(_panel([10.0] * 8))
    engine.broker = FixedFeeBroker(_costs())
    with pytest.raises(ValueError, match="cost.*NAV"):
        engine.run()


def test_forced_liquidation_cannot_charge_unpayable_fees():
    class LiquidationCostModel(CostModel):
        calls = 0

        def estimate(self, trade_value, adv_value):
            self.calls += 1
            return TradeCost(0.0 if self.calls == 1 else 101.0, 0.0, 0.0)

    engine = _engine(_panel([10.0, 10.0, np.nan, np.nan, np.nan]))
    engine.cost_model = LiquidationCostModel(_costs())
    engine.broker = BrokerSimulator(engine.cost_model)
    engine.config.delist_grace_days = 1
    with pytest.raises(ValueError, match="cost.*NAV"):
        engine.run()


def test_engine_rejects_mark_to_market_nav_overflow():
    with pytest.raises(ValueError, match="NAV.*finite"):
        _engine(_panel([10.0, 10.0, 1000.0, 1000.0, 1000.0]), initial_capital=1e308).run()


def test_engine_rejects_negative_nav():
    with pytest.raises(ValueError, match="NAV.*nonnegative"):
        _engine(_panel([10.0, 10.0, 30.0, 30.0, 30.0]), target=-1.0, allow_short=True).run()


def test_engine_preserves_zero_nav_as_a_legal_bankruptcy():
    result = _engine(_panel([10.0] * 8), cost_model=_costs(commission_bps=10_000)).run()
    assert result.equity.tolist() == pytest.approx([100.0] + [0.0] * 7)
    assert result.returns.tolist() == pytest.approx([0.0, -1.0] + [0.0] * 6)
    assert result.metrics["max_drawdown"] == pytest.approx(-1.0)
    assert np.isfinite(result.gross_equity).all()


def test_a_zero_price_wipeout_cannot_resurrect_capital():
    result = _engine(_panel([10.0, 10.0, 0.0, 20.0, 20.0])).run()
    assert result.equity.tolist() == pytest.approx([100.0, 100.0, 0.0, 0.0, 0.0])
    assert result.returns.tolist() == pytest.approx([0.0, 0.0, -1.0, 0.0, 0.0])


def test_positive_nav_cannot_be_reported_as_a_total_loss_then_resurrect():
    with pytest.raises(ValueError, match="Daily returns.*representable"):
        _engine(_panel([10.0, 10.0, 1e-20, 10.0, 10.0, 10.0])).run()

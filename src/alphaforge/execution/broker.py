"""Order generation and fill simulation.

The simulator is deliberately dumb in one respect and careful in another:

* it never invents liquidity - an order that would exceed the participation cap
  is flagged with the number of sessions it needs, and impact is charged at the
  capped rate;
* it never invents a fill - a name with no price on the execution date simply
  does not trade, and the previous weight is carried forward.  That is what
  happens on a halted or delisted name, and silently dropping it from the book
  is how backtests acquire a survivorship bias they never declared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.execution.costs import CostConfig, CostModel, total_cost_bps
from alphaforge.utils.logging import get_logger

log = get_logger("execution.broker")


@dataclass
class ExecutionResult:
    """Outcome of one rebalance: the post-trade book and what it cost."""

    weights: pd.Series  # realised post-trade weights (after costs)
    trades: pd.DataFrame  # one row per traded name
    cost_total: float
    cost_bps: float
    traded_notional: float
    unfillable: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def turnover(self) -> float:
        return float(self.trades["trade_weight"].abs().sum()) if not self.trades.empty else 0.0


@dataclass
class BrokerConfig:
    participation_cap: float = 0.10
    allow_fractional_shares: bool = True
    min_trade_value: float = 0.0  # ignore orders smaller than this
    price_source: str = "close"  # close | open | vwap

    @classmethod
    def from_dict(cls, cfg: dict | None) -> BrokerConfig:
        cfg = cfg or {}
        return cls(
            participation_cap=float(cfg.get("participation_cap", 0.10)),
            allow_fractional_shares=bool(cfg.get("allow_fractional_shares", True)),
            min_trade_value=float(cfg.get("min_trade_value", 0.0)),
            price_source=str(cfg.get("price_source", "close")),
        )


class BrokerSimulator:
    """Converts a target weight vector into trades, fills and costs."""

    def __init__(
        self,
        cost_model: CostModel | CostConfig | dict | None = None,
        config: BrokerConfig | dict | None = None,
    ) -> None:
        self.cost_model = (
            cost_model if isinstance(cost_model, CostModel) else CostModel(cost_model or {})
        )
        self.config = (
            config if isinstance(config, BrokerConfig) else BrokerConfig.from_dict(config or {})
        )

    # ------------------------------------------------------------------
    def rebalance(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series | None,
        nav: float,
        prices: pd.Series,
        adv_value: pd.Series | None = None,
    ) -> ExecutionResult:
        """Trade from ``current_weights`` to ``target_weights``.

        Parameters
        ----------
        target_weights:
            Desired weights (index = asset ids). Need not sum to one - the
            residual is cash, which is assumed to earn nothing.
        current_weights:
            Pre-trade weights. ``None`` means a cold start (nothing owned).
        nav:
            Portfolio value *before* costs.
        prices:
            Execution-date prices. A NaN price makes the name untradeable.
        adv_value:
            Average daily traded value, used for participation and impact.
        """
        cfg = self.config
        target = target_weights.astype(float)
        current = (
            current_weights.reindex(target.index).fillna(0.0).astype(float)
            if current_weights is not None
            else pd.Series(0.0, index=target.index)
        )
        current = current.reindex(target.index).fillna(0.0)

        px = prices.reindex(target.index)
        # Cash drag: the analyser wants to see it, but it is not an order.
        tradable = px.notna() & np.isfinite(px.to_numpy(dtype=float))
        unfillable = [s for s in target.index[~tradable] if abs(target[s] - current[s]) > 1e-12]

        # Untradeable names keep their current weight - we cannot sell what has
        # no price, so it stays in the book and is marked at the last close.
        # Their notional still has to be *funded*, so the tradable part of the
        # target is scaled onto the remaining budget; otherwise the book would
        # silently end up levered by the stuck position.
        effective_target = target.copy()
        budget = 1.0 - float(current[~tradable].sum())
        targeted = float(target[tradable].sum())
        if targeted > 1e-12 and budget > 0:
            effective_target[tradable] = target[tradable] * (budget / targeted)
        else:
            effective_target[tradable] = 0.0
        effective_target[~tradable] = current[~tradable]

        trade_weight = (effective_target - current).fillna(0.0)
        trade_value = trade_weight * float(nav)
        small = trade_value.abs() < cfg.min_trade_value
        trade_weight = trade_weight.where(~small, 0.0)
        trade_value = trade_weight * float(nav)

        adv = (
            adv_value.reindex(target.index)
            if adv_value is not None
            else pd.Series(np.nan, index=target.index)
        )

        active = trade_value[trade_value.abs() > 1e-9]
        if active.empty:
            return ExecutionResult(
                weights=current.copy(),
                trades=pd.DataFrame(),
                cost_total=0.0,
                cost_bps=0.0,
                traded_notional=0.0,
                unfillable=unfillable,
                diagnostics={"n_orders": 0, "n_unfillable": len(unfillable)},
            )

        costs = self.cost_model.estimate_frame(active, adv)
        participation = [
            self.cost_model.participation(v, a)
            for v, a in zip(
                active.to_numpy(dtype=float), adv.reindex(active.index).to_numpy(dtype=float)
            )
        ]
        days_needed = [
            self._days_to_complete(abs(v), a)
            for v, a in zip(
                active.to_numpy(dtype=float), adv.reindex(active.index).to_numpy(dtype=float)
            )
        ]

        trades = pd.DataFrame(
            {
                "side": np.where(active.to_numpy(dtype=float) > 0, "BUY", "SELL"),
                "price": px.reindex(active.index).to_numpy(dtype=float),
                "prev_weight": current.reindex(active.index).to_numpy(dtype=float),
                "target_weight": effective_target.reindex(active.index).to_numpy(dtype=float),
                "trade_weight": active.to_numpy(dtype=float) / float(nav),
                "trade_value": active.to_numpy(dtype=float),
                "shares": active.to_numpy(dtype=float)
                / px.reindex(active.index).to_numpy(dtype=float),
                "participation": participation,
                "days_to_complete": days_needed,
                **{k: costs[k].to_numpy(dtype=float) for k in costs.columns},
            },
            index=active.index,
        )
        if not cfg.allow_fractional_shares:
            trades["shares"] = np.trunc(trades["shares"].to_numpy(dtype=float))
            trades["trade_value"] = trades["shares"].to_numpy(dtype=float) * trades[
                "price"
            ].to_numpy(dtype=float)
            trades["trade_weight"] = trades["trade_value"].to_numpy(dtype=float) / float(nav)

        cost_total = float(trades["cost_total"].sum())
        traded_notional = float(trades["trade_value"].abs().sum())

        # Post-trade book: target weights, with costs taken out of NAV. The
        # weight vector is what the portfolio holds; the cash deduction shows up
        # in NAV, so weights are re-expressed on the smaller NAV.
        realised_weights = effective_target.copy()
        nav_after = float(nav) - cost_total
        if nav_after > 0 and abs(float(nav)) > 0:
            realised_weights = realised_weights * (float(nav) / nav_after)
        realised_weights[~tradable] = current[~tradable]

        log.debug(
            f"rebalance: {len(trades)} orders | turnover {trades['trade_weight'].abs().sum():.3f} | "
            f"cost {total_cost_bps(trades, traded_notional):.1f}bps"
        )
        return ExecutionResult(
            weights=realised_weights,
            trades=trades,
            cost_total=cost_total,
            cost_bps=total_cost_bps(trades, traded_notional),
            traded_notional=traded_notional,
            unfillable=unfillable,
            diagnostics={
                "n_orders": int(len(trades)),
                "n_unfillable": len(unfillable),
                "max_participation": float(trades["participation"].max())
                if not trades.empty
                else 0.0,
                "mean_days_to_complete": float(np.mean(days_needed)) if days_needed else 0.0,
                "commission": float(trades["commission"].sum()),
                "slippage": float(trades["slippage"].sum()),
                "impact": float(trades["impact"].sum()),
            },
        )

    # ------------------------------------------------------------------
    def _days_to_complete(self, trade_value: float, adv_value: float) -> int:
        """Sessions required to work the order inside the participation cap."""
        if not np.isfinite(adv_value) or adv_value <= 0 or trade_value <= 0:
            return 1
        return int(max(1, np.ceil(trade_value / (adv_value * self.config.participation_cap))))


__all__ = ["BrokerSimulator", "BrokerConfig", "ExecutionResult"]

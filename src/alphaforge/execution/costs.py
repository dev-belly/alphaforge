"""Transaction-cost modelling.

A backtest that ignores costs is not a backtest, it is a ranking exercise with a
currency sign in front of it.  Every cost here is charged on **traded notional**
(one-way), and the three components behave differently as size grows:

``commission``
    linear in notional - the broker's fee.
``slippage``
    linear in notional - crossing the spread plus the drift while the order is
    being worked.
``market impact``
    the square-root law: ``impact_bps = coeff * sqrt(participation)``.  Impact
    grows sub-linearly, which is exactly why a large order cannot be modelled by
    inflating slippage.  Participation is capped: an order that would exceed
    ``participation_cap`` of ADV is assumed to be worked over as many sessions
    as it takes, and the impact charged is the (higher) capped-day impact.

What is deliberately *not* modelled: the timing risk of a multi-day work (the
market can move against the un-filled remainder).  Ignoring it makes the
backtest optimistic for very large orders, and is documented as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("execution.costs")


@dataclass
class CostConfig:
    """All cost parameters live in ``configs/*.yaml`` under ``cost:``."""

    commission_bps: float = 2.0
    slippage_bps: float = 5.0
    impact_coeff_bps: float = 10.0
    participation_cap: float = 0.10
    min_commission: float = 0.0
    # Annualised borrow cost charged on short notional (0 when long-only).
    borrow_bps_annual: float = 50.0

    @classmethod
    def from_dict(cls, cfg: dict | None) -> CostConfig:
        cfg = cfg or {}
        return cls(
            commission_bps=float(cfg.get("commission_bps", 2.0)),
            slippage_bps=float(cfg.get("slippage_bps", 5.0)),
            impact_coeff_bps=float(cfg.get("impact_coeff_bps", 10.0)),
            participation_cap=float(cfg.get("participation_cap", 0.10)),
            min_commission=float(cfg.get("min_commission", 0.0)),
            borrow_bps_annual=float(cfg.get("borrow_bps_annual", 50.0)),
        )

    def to_dict(self) -> dict:
        return {
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "impact_coeff_bps": self.impact_coeff_bps,
            "participation_cap": self.participation_cap,
            "min_commission": self.min_commission,
            "borrow_bps_annual": self.borrow_bps_annual,
        }


@dataclass
class TradeCost:
    """Cost breakdown for a single order, in currency units."""

    commission: float
    slippage: float
    impact: float

    @property
    def total(self) -> float:
        return self.commission + self.slippage + self.impact


class CostModel:
    """Charges commission, slippage and square-root market impact."""

    def __init__(self, config: CostConfig | dict | None = None) -> None:
        self.config = (
            config if isinstance(config, CostConfig) else CostConfig.from_dict(config or {})
        )

    # ------------------------------------------------------------------
    def participation(self, trade_value: float, adv_value: float) -> float:
        """Fraction of a day's ADV the order represents."""
        if not np.isfinite(adv_value) or adv_value <= 0:
            return self.config.participation_cap
        return float(min(abs(trade_value) / adv_value, self.config.participation_cap))

    def impact_bps(self, participation: float) -> float:
        """Square-root impact law, in basis points of traded notional."""
        p = float(np.clip(participation, 0.0, self.config.participation_cap))
        return float(self.config.impact_coeff_bps * np.sqrt(p))

    def estimate(self, trade_value: float, adv_value: float) -> TradeCost:
        """Cost of trading ``trade_value`` (signed) against ``adv_value`` of ADV."""
        cfg = self.config
        value = abs(float(trade_value))
        if value <= 0:
            return TradeCost(0.0, 0.0, 0.0)
        participation = self.participation(value, adv_value)
        commission = max(value * cfg.commission_bps / 10_000.0, cfg.min_commission)
        slippage = value * cfg.slippage_bps / 10_000.0
        impact = value * self.impact_bps(participation) / 10_000.0
        return TradeCost(commission=commission, slippage=slippage, impact=impact)

    def estimate_frame(self, trade_values: pd.Series, adv_values: pd.Series) -> pd.DataFrame:
        """Vectorised cost estimate; returns a frame of currency costs."""
        idx = trade_values.index
        adv = adv_values.reindex(idx)
        rows = [
            self.estimate(v, a)
            for v, a in zip(trade_values.to_numpy(dtype=float), adv.to_numpy(dtype=float))
        ]
        return pd.DataFrame(
            {
                "commission": [r.commission for r in rows],
                "slippage": [r.slippage for r in rows],
                "impact": [r.impact for r in rows],
                "cost_total": [r.total for r in rows],
            },
            index=idx,
        )

    def borrow_cost(self, short_notional: float, days: int = 1) -> float:
        """Financing charged on short exposure, pro-rated by calendar time."""
        if short_notional <= 0:
            return 0.0
        return float(short_notional * self.config.borrow_bps_annual / 10_000.0 * days / 252.0)


def total_cost_bps(costs: pd.DataFrame, traded_notional: float) -> float:
    """Total cost expressed in bps of traded notional (the comparable number)."""
    if traded_notional <= 0:
        return 0.0
    return float(costs["cost_total"].sum() / traded_notional * 10_000.0)


__all__ = ["CostConfig", "CostModel", "TradeCost", "total_cost_bps"]

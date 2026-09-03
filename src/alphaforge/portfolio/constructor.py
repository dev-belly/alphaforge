"""Glue between alpha scores, the risk model and the optimiser.

The constructor answers one question per rebalance date: *given the scores, the
covariance and the constraints, what should the book look like?*  It is the only
place that knows how a score becomes an expected return, so the backtester can
stay a pure accounting engine.

Information coefficient
-----------------------
``implied_expected_returns`` needs an IC.  The value used here is always the
**walk-forward out-of-sample** IC produced by the model layer - never an
in-sample fit - and it is shrunk toward zero because an IC estimated on a few
hundred cross-sections is itself noisy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.features.panel import MarketPanel
from alphaforge.portfolio.expected_returns import implied_expected_returns
from alphaforge.portfolio.optimizer import OptimizationResult, OptimizerConfig, PortfolioOptimizer
from alphaforge.risk.covariance import CovarianceEstimator
from alphaforge.utils.logging import get_logger

log = get_logger("portfolio.constructor")


@dataclass
class ConstructionConfig:
    """Covariance / alpha-scaling settings for the constructor."""

    covariance_method: str = "ledoit_wolf"
    covariance_lookback: int = 252
    min_observations: int = 60
    ewma_halflife: int = 90
    ic_shrinkage: float = 0.5
    volatility_targeting: bool = True

    @classmethod
    def from_dict(cls, cfg: dict | None) -> ConstructionConfig:
        risk = dict((cfg or {}).get("risk", {}) or {})
        return cls(
            covariance_method=str(risk.get("covariance_method", "ledoit_wolf")),
            covariance_lookback=int(risk.get("covariance_lookback", 252)),
            min_observations=int(risk.get("min_observations", 60)),
            ewma_halflife=int(risk.get("ewma_halflife", 90)),
            ic_shrinkage=float((cfg or {}).get("ic_shrinkage", 0.5)),
            volatility_targeting=bool((cfg or {}).get("volatility_targeting", True)),
        )


class PortfolioConstructor:
    """Turns a cross-section of scores into constrained target weights."""

    def __init__(
        self,
        panel: MarketPanel,
        optimizer_config: OptimizerConfig | dict | None = None,
        construction_config: ConstructionConfig | dict | None = None,
    ) -> None:
        self.panel = panel
        self.optimizer_config = (
            optimizer_config
            if isinstance(optimizer_config, OptimizerConfig)
            else OptimizerConfig.from_dict(optimizer_config or {})
        )
        self.construction_config = (
            construction_config
            if isinstance(construction_config, ConstructionConfig)
            else ConstructionConfig.from_dict(construction_config or {})
        )
        self.optimizer = PortfolioOptimizer(self.optimizer_config)
        self._cov_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    def eligible(self, date: pd.Timestamp) -> pd.Index:
        """Names that are investable and have enough return history on ``date``."""
        pos = self.panel.dates.get_loc(date)
        start = max(0, pos - self.construction_config.covariance_lookback)
        window = self.panel.returns.iloc[start:pos]
        counts = window.notna().sum()
        tradable = self.panel.universe.loc[date]
        tradable = tradable[tradable.astype(bool)] if hasattr(tradable, "astype") else tradable
        keep = counts[counts >= self.construction_config.min_observations].index
        return pd.Index(sorted(set(keep) & set(tradable.index)))

    def covariance(self, date: pd.Timestamp, assets: pd.Index | None = None) -> pd.DataFrame:
        """Annualised covariance estimated on the trailing window only."""
        cached = self._cov_cache.get(date)
        if cached is not None and (assets is None or list(cached.columns) == list(assets)):
            return cached
        assets = assets if assets is not None else self.eligible(date)
        pos = self.panel.dates.get_loc(date)
        start = max(0, pos - self.construction_config.covariance_lookback)
        window = self.panel.returns.iloc[start:pos][list(assets)].dropna(axis=1, how="all")
        if window.shape[1] < 2:
            raise ValueError(f"Not enough return history to estimate risk on {date.date()}")
        est = CovarianceEstimator(
            self.construction_config.covariance_method,
            halflife=self.construction_config.ewma_halflife,
        ).estimate(window)
        self._cov_cache[date] = est.matrix
        return est.matrix

    def volatility(self, date: pd.Timestamp, assets: pd.Index) -> pd.Series:
        cov = self.covariance(date, assets)
        vol = pd.Series(
            np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), 1e-12, None)), index=cov.columns
        )
        return vol.reindex(assets).fillna(vol.median())

    def industry(self, date: pd.Timestamp, assets: pd.Index) -> pd.Series | None:
        if self.panel.industry is None or self.panel.industry.empty:
            return None
        row = self.panel.industry.loc[date] if date in self.panel.industry.index else None
        if row is None:
            return None
        return row.reindex(assets).astype("object")

    # ------------------------------------------------------------------
    def construct(
        self,
        date: pd.Timestamp,
        scores: pd.Series,
        prev_weights: pd.Series | None = None,
        ic: float = 0.05,
        benchmark_weights: pd.Series | None = None,
    ) -> OptimizationResult:
        """Full construction step for one rebalance date."""
        assets = self.eligible(date)
        scores = scores.reindex(assets).dropna()
        if len(scores) < max(self.optimizer_config.min_names, 5):
            raise ValueError(
                f"Only {len(scores)} scored names on {date.date()} - skipping rebalance"
            )

        assets = pd.Index(scores.index)
        cov = self.covariance(date, assets)
        vol = self.volatility(date, assets)
        mu = implied_expected_returns(
            scores=scores,
            volatility=vol,
            ic=ic,
            ic_shrinkage=self.construction_config.ic_shrinkage,
        )

        result = self.optimizer.solve(
            mu=mu,
            cov=cov,
            prev_weights=prev_weights,
            industry=self.industry(date, assets),
            benchmark_weights=benchmark_weights,
        )

        if (
            self.construction_config.volatility_targeting
            and self.optimizer_config.target_volatility
        ):
            result = self._apply_vol_target(result, cov)
        result.diagnostics["ic_used"] = float(ic)
        result.diagnostics["n_scored"] = int(len(scores))
        return result

    # ------------------------------------------------------------------
    def _apply_vol_target(
        self, result: OptimizationResult, cov: pd.DataFrame
    ) -> OptimizationResult:
        """Scale the book down to the volatility budget when the QP could not.

        The constraint is in the QP, but with a binding turnover or industry
        constraint the solver can return a hotter book than the budget allows;
        de-levering into cash is the honest fallback.
        """
        assert self.optimizer_config.target_volatility is not None
        target = float(self.optimizer_config.target_volatility)
        realised = float(result.diagnostics.get("ex_ante_vol", 0.0))
        if realised <= target or realised <= 0:
            return result
        scale = target / realised
        w = result.weights * scale
        result.weights = w
        result.diagnostics["vol_target_scale"] = scale
        result.diagnostics["ex_ante_vol"] = float(
            np.sqrt(
                max(
                    w.to_numpy(dtype=float)
                    @ cov.reindex(index=w.index, columns=w.index).to_numpy(dtype=float)
                    @ w.to_numpy(dtype=float),
                    0.0,
                )
            )
        )
        result.diagnostics["cash_weight"] = float(1.0 - w.sum())
        return result


__all__ = ["PortfolioConstructor", "ConstructionConfig"]

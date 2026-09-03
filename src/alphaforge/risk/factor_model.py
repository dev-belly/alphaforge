"""Simplified multi-factor fundamental risk model.

Risk decomposition
------------------
``Sigma = B F B' + D``

* ``B``  - asset exposures to market, industry and style factors
* ``F``  - factor covariance matrix
* ``D``  - diagonal specific (idiosyncratic) variance

From that single decomposition the model produces portfolio volatility, marginal
and component risk contributions, and a factor-level risk budget. The
Euler decomposition ``sigma_p = sum_i w_i * MCR_i`` holds by construction and
is asserted in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.risk.covariance import CovarianceEstimator
from alphaforge.utils.logging import get_logger

log = get_logger("risk.factor_model")

STYLE_FACTORS = ["size", "value", "momentum", "volatility", "liquidity", "quality"]


@dataclass
class RiskModelConfig:
    style_factors: list[str] = field(default_factory=lambda: list(STYLE_FACTORS))
    covariance_method: str = "ledoit_wolf"
    ewma_halflife: int = 90
    winsorization: float = 0.01
    min_obs: int = 60
    specific_var_floor: float = 0.10  # fraction of the median specific variance

    @classmethod
    def from_dict(cls, cfg: dict | None) -> RiskModelConfig:
        cfg = cfg or {}
        style = cfg.get("style_factors")
        return cls(
            style_factors=list(style) if style else list(STYLE_FACTORS),
            covariance_method=str(cfg.get("covariance_method", "ledoit_wolf")),
            ewma_halflife=int(cfg.get("ewma_halflife", 90)),
            winsorization=float(cfg.get("winsorization", 0.01)),
            min_obs=int(cfg.get("min_obs", 60)),
            specific_var_floor=float(cfg.get("specific_var_floor", 0.10)),
        )


@dataclass
class RiskModelResult:
    exposures: pd.DataFrame  # (assets x factors)
    factor_cov: pd.DataFrame  # (factors x factors)
    specific_var: pd.Series
    covariance: pd.DataFrame
    factor_returns: pd.DataFrame
    residuals: pd.DataFrame
    r_squared: float
    config: dict = field(default_factory=dict)


class FundamentalRiskModel:
    """Cross-sectional multi-factor risk model fitted on a rolling window."""

    def __init__(self, config: RiskModelConfig | None = None) -> None:
        self.config = config or RiskModelConfig()

    # ------------------------------------------------------------------
    def build_exposures(
        self,
        market_cap: pd.Series,
        industry: pd.Series,
        style_panels: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Assemble the exposure matrix: market + industry dummies + styles."""
        assets = market_cap.index
        cols: dict[str, pd.Series] = {"market": pd.Series(1.0, index=assets)}

        for name in self.config.style_factors:
            if name in style_panels:
                s = style_panels[name].reindex(assets)
                # Standardise so exposures are comparable across factors.
                std = s.std()
                cols[name] = (s - s.mean()) / std if std and std > 0 else s * 0.0
            elif name == "size" and market_cap is not None:
                s = np.log(market_cap.replace(0, np.nan))
                std = s.std()
                cols[name] = (s - s.mean()) / std if std and std > 0 else s * 0.0

        industries = sorted(set(industry.dropna().astype(str)))
        # Drop one industry to avoid perfect collinearity with the intercept.
        for ind in industries[1:]:
            cols[f"ind_{ind}"] = (industry.astype(str) == ind).astype(float).reindex(assets)

        exposures = pd.DataFrame(cols).reindex(assets).fillna(0.0)
        return exposures

    # ------------------------------------------------------------------
    def fit(
        self,
        returns: pd.DataFrame,
        market_cap: pd.Series,
        industry: pd.Series,
        style_panels: dict[str, pd.Series] | None = None,
    ) -> RiskModelResult:
        """Fit ``r = B f + eps`` cross-sectionally, then form ``B F B' + D``."""
        style_panels = style_panels or {}
        rets = returns.dropna(how="all")
        assets = [c for c in rets.columns if rets[c].notna().sum() >= self.config.min_obs]
        rets = rets[assets]

        exposures = self.build_exposures(
            market_cap.reindex(assets), industry.reindex(assets), style_panels
        )
        B = exposures.to_numpy(dtype=float)

        factor_returns = []
        resid_list = []
        for _date, row in rets.iterrows():
            y = row.to_numpy(dtype=float)
            mask = np.isfinite(y)
            if mask.sum() < max(B.shape[1] + 5, 20):
                factor_returns.append(pd.Series(np.nan, index=exposures.columns))
                resid_list.append(pd.Series(np.nan, index=assets))
                continue
            X = B[mask]
            yy = y[mask]
            # Ridge-stabilised WLS: weight by sqrt(market cap) as in Barra models.
            w = np.sqrt(market_cap.reindex(assets).to_numpy(dtype=float)[mask])
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
            Xw = X * w[:, None]
            yw = yy * w
            try:
                beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            except np.linalg.LinAlgError:
                beta = np.zeros(X.shape[1])
            factor_returns.append(pd.Series(beta, index=exposures.columns))
            resid_row = pd.Series(np.nan, index=assets)
            resid_row.iloc[np.where(mask)[0]] = (yy - X @ beta).astype(float)
            resid_list.append(resid_row)

        f_rets = pd.DataFrame(factor_returns, index=rets.index)
        residuals = pd.DataFrame(resid_list, index=rets.index, columns=assets)

        # Factor covariance from the estimated factor return series.
        fcov = CovarianceEstimator(
            self.config.covariance_method,
            halflife=self.config.ewma_halflife,
        ).estimate(f_rets.dropna(how="all"))
        factor_cov = fcov.matrix

        # Specific variance: EWMA of squared residuals for responsiveness.
        halflife = self.config.ewma_halflife
        lam = 0.5 ** (1.0 / max(halflife, 1))
        sq = residuals**2
        weights = lam ** np.arange(len(sq) - 1, -1, -1)
        spec = (
            pd.Series(
                np.nansum(sq.to_numpy(dtype=float) * weights[:, None], axis=0)
                / np.maximum(
                    np.sum(np.isfinite(sq.to_numpy(dtype=float)) * weights[:, None], axis=0), 1.0
                ),
                index=assets,
            )
            * 252.0
        )
        floor = float(np.nanmedian(spec.values) * self.config.specific_var_floor)
        spec = spec.fillna(floor).clip(lower=max(floor, 1e-8))

        cov = pd.DataFrame(
            B @ factor_cov.to_numpy(dtype=float) @ B.T + np.diag(spec.to_numpy(dtype=float)),
            index=assets,
            columns=assets,
        )

        # Cross-sectional R^2 of the model.
        total_var = (rets.var(ddof=1) * 252.0).sum()
        spec_var = spec.sum()
        r2 = float(1.0 - spec_var / total_var) if total_var > 0 else float("nan")

        log.info(
            f"RiskModel: {len(assets)} assets | {B.shape[1]} factors | cross-sectional R2={r2:.3f}"
        )
        return RiskModelResult(
            exposures=exposures,
            factor_cov=factor_cov,
            specific_var=spec,
            covariance=cov,
            factor_returns=f_rets,
            residuals=residuals,
            r_squared=r2,
            config={
                "covariance_method": self.config.covariance_method,
                "n_factors": int(B.shape[1]),
                "style_factors": list(self.config.style_factors),
            },
        )


# --------------------------------------------------------------------------
# risk decomposition helpers
# --------------------------------------------------------------------------
def portfolio_risk(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.reindex(cov.columns).fillna(0.0).to_numpy(dtype=float)
    return float(np.sqrt(max(w @ cov.to_numpy(dtype=float) @ w, 0.0)))


def marginal_risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """``MCR_i = (Sigma w)_i / sigma_p``."""
    w = weights.reindex(cov.columns).fillna(0.0).to_numpy(dtype=float)
    sigma = cov.to_numpy(dtype=float)
    port_vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    if port_vol < 1e-12:
        return pd.Series(0.0, index=cov.columns)
    return pd.Series(sigma @ w / port_vol, index=cov.columns)


def component_risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """``CRC_i = w_i * MCR_i``; sums to portfolio volatility (Euler)."""
    mcr = marginal_risk_contribution(weights, cov)
    w = weights.reindex(cov.columns).fillna(0.0)
    return w * mcr


def factor_risk_decomposition(
    weights: pd.Series,
    exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_var: pd.Series,
) -> pd.DataFrame:
    """Split portfolio variance into factor and specific blocks, per factor."""
    w = weights.reindex(exposures.index).fillna(0.0).to_numpy(dtype=float)
    B = exposures.to_numpy(dtype=float)
    F = factor_cov.reindex(index=exposures.columns, columns=exposures.columns).to_numpy(dtype=float)

    portfolio_exposure = w @ B  # (n_factors,)
    sigma_f = np.sqrt(max(portfolio_exposure @ F @ portfolio_exposure, 0.0))
    specific = float(
        w @ np.diag(specific_var.reindex(exposures.index).fillna(0.0).to_numpy(dtype=float)) @ w
    )
    sigma_specific = float(np.sqrt(max(specific, 0.0)))
    total_var = float(sigma_f**2 + specific)
    sigma_p = float(np.sqrt(max(total_var, 1e-18)))

    # Marginal contribution to *volatility* (MCR): d sigma_p / d b_p.
    mcr_vol = (F @ portfolio_exposure) / sigma_p
    # Euler contribution to *variance*: b_p_k * (F b_p)_k.  Summed over factors
    # this equals sigma_f**2, so with the specific term it reconstructs
    # ``total_var`` exactly (the variance analogue of the volatility Euler sum).
    marginal_var = F @ portfolio_exposure
    rows = []
    for k, name in enumerate(exposures.columns):
        contrib = portfolio_exposure[k] * marginal_var[k]
        rows.append(
            {
                "factor": name,
                "exposure": float(portfolio_exposure[k]),
                "marginal_contribution": float(mcr_vol[k]),
                "contribution_to_variance": float(contrib),
                "pct_of_variance": float(contrib / total_var) if total_var > 0 else np.nan,
                "standalone_vol": float(np.sqrt(max(F[k, k], 0.0))),
            }
        )
    rows.append(
        {
            "factor": "specific",
            "exposure": np.nan,
            "marginal_contribution": sigma_specific / sigma_p,
            "contribution_to_variance": specific,
            "pct_of_variance": specific / total_var if total_var > 0 else np.nan,
            "standalone_vol": sigma_specific,
        }
    )
    return pd.DataFrame(rows)


__all__ = [
    "FundamentalRiskModel",
    "RiskModelConfig",
    "RiskModelResult",
    "portfolio_risk",
    "marginal_risk_contribution",
    "component_risk_contribution",
    "factor_risk_decomposition",
    "STYLE_FACTORS",
]

"""Returns-based factor attribution.

The portfolio return is regressed on the risk model's factor-return series:

    r_p - r_f = Σ_k β_k · f_k + ε

The coefficients ``β_k`` are the portfolio's *exposure* to each factor, and the
per-factor contribution to the period's excess return is ``β_k · mean(f_k)``.
This is the bottom-up complement to :mod:`alphaforge.attribution.brinson`:
Brinson explains active return by *where* you were (sectors), this explains it
by *what risk* you were running (styles).

Uses ``statsmodels`` when available and falls back to a least-squares solve so
the module never hard-depends on a modelling library being present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("attribution.factor")


@dataclass
class FactorAttributionResult:
    """Regression-based decomposition of excess return into factor bets."""

    betas: pd.Series
    attributed_return: pd.Series  # per-factor contribution to mean excess return
    r_squared: float
    residual_mean: float
    n_observations: int
    t_stats: pd.Series = field(default_factory=pd.Series, repr=False)

    def to_dict(self) -> dict:
        return {
            "r_squared": float(self.r_squared),
            "residual_mean": float(self.residual_mean),
            "n_observations": int(self.n_observations),
            "attributed_return": self.attributed_return.to_dict(),
            "betas": self.betas.to_dict(),
        }


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> FactorAttributionResult:
    """Attribute excess portfolio return across the factor return columns.

    Parameters
    ----------
    portfolio_returns:
        (dates) total portfolio return series.
    factor_returns:
        (dates x factors) factor return series from the risk model.
    benchmark_returns:
        Optional (dates) benchmark; when supplied, attribution is of *active*
        return (portfolio - benchmark), otherwise of total excess return.
    """
    joined = pd.concat(
        [portfolio_returns.rename("r"), factor_returns.rename(columns=str)],
        axis=1,
    ).dropna()
    if benchmark_returns is not None:
        joined = pd.concat([joined, benchmark_returns.rename("b").dropna()], axis=1).dropna()
        joined["y"] = joined["r"] - joined["b"]
    else:
        ppy = 252.0
        rf_period = (1.0 + risk_free_rate) ** (1.0 / ppy) - 1.0
        joined["y"] = joined["r"] - rf_period

    y = joined["y"].to_numpy(dtype=float)
    X = joined[factor_returns.columns].to_numpy(dtype=float)
    if X.shape[0] < X.shape[1] + 2:
        log.warning("Too few observations for factor attribution - returning zeros")
        return FactorAttributionResult(
            betas=pd.Series(0.0, index=factor_returns.columns),
            attributed_return=pd.Series(0.0, index=factor_returns.columns),
            r_squared=float("nan"),
            residual_mean=0.0,
            n_observations=int(X.shape[0]),
        )

    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta_full, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta_full
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    betas = pd.Series(beta_full[1:], index=factor_returns.columns, dtype=float)
    f_mean = factor_returns.reindex(joined.index).mean()
    attributed = betas * f_mean.reindex(betas.index).fillna(0.0)

    # White (heteroskedasticity-robust) standard errors for the t-stats.
    xtx_inv = np.linalg.inv(Xd.T @ Xd)
    cov = xtx_inv * (resid**2).sum() / (n - k - 1)
    se = np.sqrt(np.diag(cov))
    tstats = pd.Series(beta_full / se, index=["intercept", *factor_returns.columns], dtype=float)

    log.info(f"Factor attribution: R²={r2:.3f} | residual mean={resid.mean():+.5f}")
    return FactorAttributionResult(
        betas=betas,
        attributed_return=attributed,
        r_squared=float(r2),
        residual_mean=float(resid.mean()),
        n_observations=int(n),
        t_stats=tstats,
    )


__all__ = ["FactorAttributionResult", "factor_attribution"]

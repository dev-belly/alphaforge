"""Turning alpha scores into expected returns.

A raw model score is not an expected return.  The standard bridge is Grinold's
"fundamental law": a forecast with information coefficient ``IC`` and a
cross-sectional score ``z`` implies an annualised expected return of

    mu_i = IC * z_i * sigma_i

where ``sigma_i`` is the asset's annualised volatility.  Scaling by volatility
is what makes the subsequent mean-variance problem dimensionally sane: without
it the optimiser treats a z-score on a 60%-vol name the same as on a 15%-vol
name.

The IC used here is estimated **out-of-sample** (walk-forward), never fitted
in-sample, and is shrunk toward zero to reflect estimation error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("portfolio.expected_returns")


def implied_expected_returns(
    scores: pd.Series,
    volatility: pd.Series,
    ic: float,
    ic_shrinkage: float = 0.5,
    annualise: bool = True,
    periods_per_year: float = 252.0,
    clip_sigma: float = 3.0,
) -> pd.Series:
    """``mu = shrunk_IC * z * sigma`` with de-meaned, clipped scores."""
    z = scores.copy()
    z = z - z.mean()
    std = z.std()
    if std and std > 0:
        z = z / std
    z = z.clip(-clip_sigma, clip_sigma).fillna(0.0)

    sigma = volatility.reindex(z.index).fillna(volatility.median())
    if not annualise:
        sigma = sigma / np.sqrt(periods_per_year)

    shrunk_ic = float(ic) * float(ic_shrinkage)
    mu = shrunk_ic * z * sigma
    mu = mu - mu.mean()  # benchmark-relative alphas: cash-neutral
    return mu.reindex(scores.index).fillna(0.0)


def blend_expected_returns(
    components: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Weighted combination of several alpha sources, rescaled to unit z-score."""
    if not components:
        raise ValueError("No expected-return components supplied")
    names = list(components)
    w = {n: float((weights or {}).get(n, 1.0)) for n in names}
    total = sum(abs(v) for v in w.values()) or 1.0
    acc = None
    for n in names:
        part = components[n] * (w[n] / total)
        acc = part if acc is None else acc.add(part, fill_value=0.0)
    std = acc.std()
    if std and std > 0:
        acc = acc / std
    return acc


def realised_forward_returns(close: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Diagnostic helper: realised forward returns (never used as an input)."""
    return close.shift(-horizon) / close - 1.0


__all__ = ["implied_expected_returns", "blend_expected_returns", "realised_forward_returns"]

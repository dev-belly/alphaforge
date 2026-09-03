"""Cross-sectional factor preprocessing.

Every transform is applied **within a single date** - a factor value for date
``t`` is never standardised, winsorised or neutralised using observations from
any other date.

Industry + size neutralisation follows the Frisch-Waugh-Lovell theorem, so the
sequential implementation below is algebraically identical to a single joint OLS
of the factor on industry dummies *and* log market cap:

    resid = M_D y - (M_D s) * cov(M_D y, M_D s) / var(M_D s)

where ``M_D`` is the within-industry demeaning operator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.factors.base import Factor, FactorContext
from alphaforge.utils.logging import get_logger

log = get_logger("factors.preprocess")


@dataclass
class ProcessingConfig:
    """Mirrors the ``factor`` block of ``configs/default.yaml``."""

    winsorize: bool = True
    winsorization: float = 0.01
    standardize: bool = True
    rank_transform: bool = False
    industry_neutralize: bool = True
    size_neutralize: bool = True
    demean: bool = True
    min_names: int = 5
    fill_value: float = 0.0

    @classmethod
    def from_dict(cls, cfg: dict) -> ProcessingConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (cfg or {}).items() if k in known})


def group_mean(df: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
    """Efficient within-group (industry) mean broadcast back to the column space."""
    arr = df.to_numpy(dtype=float)
    labels = mapping.reindex(df.columns).to_numpy()
    out = np.full(arr.shape, np.nan)
    uniq = [u for u in pd.unique(labels) if isinstance(u, str)]
    if not uniq:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(arr, axis=1)
        return pd.DataFrame(
            np.tile(means[:, None], (1, arr.shape[1])), index=df.index, columns=df.columns
        )
    for u in uniq:
        cols = np.where(labels == u)[0]
        if cols.size == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(arr[:, cols], axis=1)
        out[:, cols] = m[:, None]
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def winsorize_panel(df: pd.DataFrame, lower: float = 0.01) -> pd.DataFrame:
    """Cross-sectional two-sided winsorization at the given tail quantiles."""
    if lower <= 0:
        return df
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(1.0 - lower, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def standardize_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score (NaN-preserving)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=1).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def rank_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks in [0, 1]."""
    return df.rank(axis=1, pct=True)


def demean_panel(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0)


def neutralize_continuous(y: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Residualise ``y`` on ``s`` separately for every date (with intercept)."""
    ym = y.sub(y.mean(axis=1), axis=0)
    sm = s.sub(s.mean(axis=1), axis=0)
    cov = (ym * sm).mean(axis=1)
    var = (sm**2).mean(axis=1)
    beta = cov / var.replace(0.0, np.nan)
    return ym.sub(sm.mul(beta, axis=0), fill_value=0.0)


class FactorPreprocessor:
    """Applies the configured preprocessing chain to a factor panel."""

    def __init__(self, config: ProcessingConfig, ctx: FactorContext) -> None:
        self.config = config
        self.ctx = ctx
        self._industry = (
            ctx.panel.industry.ffill().iloc[-1] if len(ctx.panel.industry) else pd.Series(dtype=str)
        )
        self._size = np.log(ctx.panel.market_cap.replace(0.0, np.nan))

    # ------------------------------------------------------------------
    def process(self, factor: Factor, apply_universe: bool = True) -> pd.DataFrame:
        cfg = self.config
        df = factor.raw.astype(float)
        if apply_universe:
            df = df.where(self.ctx.panel.universe)

        # Too few cross-sectional observations => the transform is not meaningful.
        valid = df.notna().sum(axis=1) >= cfg.min_names
        df = df.where(valid)

        if cfg.winsorize:
            df = winsorize_panel(df, cfg.winsorization)

        # Orient the factor so that higher = higher expected return.
        if factor.spec.direction == -1:
            df = -df

        if cfg.industry_neutralize and len(self._industry):
            ind_mean = group_mean(df, self._industry)
            if cfg.size_neutralize:
                size_demeaned = self._size - group_mean(self._size, self._industry)
                y_perp = df - ind_mean
                s_perp = size_demeaned
                y_perp_m = y_perp.sub(y_perp.mean(axis=1), axis=0)
                s_perp_m = s_perp.sub(s_perp.mean(axis=1), axis=0)
                cov = (y_perp_m * s_perp_m).mean(axis=1)
                var = (s_perp_m**2).mean(axis=1)
                beta = cov / var.replace(0.0, np.nan)
                df = y_perp_m.sub(s_perp_m.mul(beta, axis=0), fill_value=0.0) + 0.0
            else:
                df = df - ind_mean
        elif cfg.size_neutralize:
            df = neutralize_continuous(df, self._size)
        elif cfg.demean:
            df = demean_panel(df)

        if cfg.rank_transform:
            df = rank_panel(df)
        if cfg.standardize:
            df = standardize_panel(df)

        # Neutral exposure for names we cannot score, so portfolio construction
        # stays fully invested instead of dropping half the universe.
        return df.fillna(cfg.fill_value).where(valid, np.nan)

    # ------------------------------------------------------------------
    def process_many(self, factors: dict[str, Factor]) -> dict[str, pd.DataFrame]:
        return {name: self.process(f) for name, f in factors.items()}


__all__ = [
    "ProcessingConfig",
    "FactorPreprocessor",
    "winsorize_panel",
    "standardize_panel",
    "rank_panel",
    "demean_panel",
    "neutralize_continuous",
    "group_mean",
]

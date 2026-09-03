"""Market-regime classification for cross-sectional quant research.

Each trading day is labelled along two independent axes inferred *only* from
information available as of that date (rolling windows, never the future):

* **trend**  : Bull / Bear -- is the cumulative index above or below its
  trailing moving average?
* **volatility** : High / Low -- is the rolling annualised volatility above or
  below its own historical median?

The four combined labels are
``Bull/LowVol``, ``Bull/HighVol``, ``Bear/LowVol``, ``Bear/HighVol``.

This is a research aid, not a trading signal.  It is fully reproducible and is
used to study how factors and the portfolio behave in different environments
(see Master Prompt, "Market Regime Analysis").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.utils.math_utils import infer_periods_per_year

REGIME_LABELS = ["Bull/LowVol", "Bull/HighVol", "Bear/LowVol", "Bear/HighVol"]


def classify_regime(
    market_returns: pd.Series,
    vol_window: int = 63,
    trend_window: int = 126,
    vol_quantile: float = 0.5,
) -> pd.Series:
    """Per-date regime label.

    Parameters
    ----------
    market_returns:
        Daily market / benchmark return series (used for vol + trend).
    vol_window:
        Trading days for the rolling annualised-volatility estimate.
    trend_window:
        Trading days for the cumulative-return vs MA trend comparison.
    vol_quantile:
        Fraction of history used as the High/Low volatility threshold
        (0.5 = own median).

    Returns
    -------
    pd.Series of str labels, indexed like ``market_returns``; ``pd.NA`` where
    there is not enough history to decide.
    """
    r = market_returns.dropna().astype(float)
    min_hist = max(vol_window, trend_window) + 1
    if len(r) < min_hist:
        return pd.Series(index=r.index, dtype=object)

    ppy = infer_periods_per_year(pd.DatetimeIndex(r.index))
    rolling_vol = r.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(ppy)
    vol_threshold = float(rolling_vol.quantile(vol_quantile))

    cum = (1.0 + r).cumprod()
    trend_ma = cum.rolling(trend_window, min_periods=trend_window).mean()
    trend = cum - trend_ma  # > 0 bull, < 0 bear

    high_vol = rolling_vol > vol_threshold
    bull = trend > 0
    labels = np.where(bull.to_numpy(), "Bull/", "Bear/") + np.where(
        high_vol.to_numpy(), "HighVol", "LowVol"
    )
    out = pd.Series(labels, index=r.index, dtype=object)
    undecided = rolling_vol.isna() | trend.isna()
    out[undecided] = pd.NA
    return out


def regime_statistics(returns: pd.Series, regime: pd.Series) -> dict:
    """Annualised return / vol / Sharpe of ``returns`` per regime label."""
    r = returns.dropna()
    reg = regime.reindex(r.index)
    stats: dict[str, dict] = {}
    for lab in REGIME_LABELS:
        mask = reg == lab
        if int(mask.sum()) < 5:
            continue
        rs = r[mask]
        ppy = infer_periods_per_year(pd.DatetimeIndex(rs.index))
        ann_ret = float((1.0 + rs).prod() ** (ppy / len(rs)) - 1.0) if len(rs) else float("nan")
        ann_vol = float(rs.std(ddof=1) * np.sqrt(ppy))
        sharpe = float(rs.mean() * ppy / ann_vol) if ann_vol > 0 else float("nan")
        stats[lab] = {
            "n_days": int(mask.sum()),
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
        }
    return stats


def factor_performance_by_regime(ic_series: pd.Series, regime: pd.Series) -> dict:
    """Rank-IC diagnostics of one factor, split by market regime.

    For each regime label returns ``n``, ``ic_mean``, ``icir`` (mean / std of
    IC) and ``positive_ic_ratio``.
    """
    ic = ic_series.dropna()
    reg = regime.reindex(ic.index)
    out: dict[str, dict] = {}
    for lab in REGIME_LABELS:
        mask = reg == lab
        x = ic[mask]
        if len(x) < 5:
            continue
        ic_mean = float(x.mean())
        ic_std = float(x.std(ddof=1))
        icir = ic_mean / ic_std if ic_std > 0 else float("nan")
        out[lab] = {
            "n": int(len(x)),
            "ic_mean": ic_mean,
            "icir": icir,
            "positive_ic_ratio": float((x > 0).mean()),
        }
    return out


__all__ = [
    "REGIME_LABELS",
    "classify_regime",
    "regime_statistics",
    "factor_performance_by_regime",
]

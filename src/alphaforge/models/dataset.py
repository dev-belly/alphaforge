"""Dataset construction for cross-sectional alpha models.

Target definition
-----------------
The task is **cross-sectional expected-return estimation**, not "will the stock
go up tomorrow".  Two supported targets:

``forward_return``
    Realised forward ``horizon``-day return, cross-sectionally rank-transformed
    to ``[-0.5, 0.5]`` within each date.  Ranking removes the market's common
    time-series drift and turns the regression into a ranking problem, which is
    what a long-short portfolio actually trades.

``forward_rank``
    Percentile rank of the forward return, identical information to the above
    but bounded in ``[0, 1]``.

Leakage guards
--------------
* features and labels are aligned so the feature at ``t`` only uses data up to
  ``t`` (guaranteed by the factor layer);
* ``forward_returns`` is shifted by ``-horizon``, so the label is only knowable
  after the holding period;
* rows whose label is NaN (end of sample, delisted names) are dropped from
  training but retained for prediction diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.features.panel import MarketPanel
from alphaforge.utils.logging import get_logger

log = get_logger("models.dataset")


@dataclass
class AlphaDataset:
    """Long-format supervised dataset for cross-sectional alpha modelling."""

    features: pd.DataFrame  # (n_samples x n_features)
    target: pd.Series
    dates: pd.Series
    symbols: pd.Series
    forward_returns: pd.Series  # raw label, kept for portfolio-level evaluation
    feature_names: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.target)

    def date_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(pd.unique(self.dates)))

    def mask_by_dates(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return self.dates.isin(set(dates)).to_numpy()

    def describe(self) -> dict:
        return {
            "n_samples": len(self),
            "n_features": len(self.feature_names),
            "n_dates": int(self.dates.nunique()),
            "n_symbols": int(self.symbols.nunique()),
            "start": str(pd.Timestamp(self.dates.min()).date()),
            "end": str(pd.Timestamp(self.dates.max()).date()),
            "target": self.metadata.get("target_type", "forward_rank"),
        }


def build_dataset(
    panel: MarketPanel,
    factor_panels: dict[str, pd.DataFrame],
    horizon: int = 21,
    target: str = "forward_rank",
    universe: pd.DataFrame | None = None,
    min_names_per_date: int = 20,
    dropna_labels: bool = True,
) -> AlphaDataset:
    """Stack factor panels into the supervised long format."""
    if not factor_panels:
        raise ValueError("No factor panels supplied")

    uni = panel.universe if universe is None else universe
    fwd = (panel.close.shift(-horizon) / panel.close - 1.0).where(uni)
    dates = panel.dates
    symbols = panel.symbols

    # Build the long frame in one allocation. Stacking 42 panels of ~400k rows
    # individually and concatenating them costs several GB; ravel()-ing each
    # panel straight into a pre-sized dict does not.
    long_index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    data = {
        name: np.asarray(
            fp.reindex(index=dates, columns=symbols).where(uni).to_numpy(dtype=float)
        ).ravel()
        for name, fp in factor_panels.items()
    }
    names = list(data)
    data["fwd_return"] = np.asarray(
        fwd.reindex(index=dates, columns=symbols).to_numpy(dtype=float)
    ).ravel()

    df = pd.DataFrame(data, index=long_index, copy=False)
    df = df.reset_index()
    del data, long_index

    # Cross-sectional labels: percentile rank of the forward return within date.
    grp = df.groupby("date")["fwd_return"]
    df["target"] = grp.transform(lambda s: s.rank(pct=True) - 0.5)
    if target not in {"forward_rank", "forward_return"}:
        df["target"] = df["fwd_return"]  # raw-return diagnostic mode
    df.loc[df["fwd_return"].isna(), "target"] = np.nan

    counts = df.groupby("date")["symbol"].transform("size")
    df = df[counts >= min_names_per_date]
    if dropna_labels:
        df = df[df["target"].notna()]

    # A NaN feature means "no signal today" -> neutral (zero) exposure. This is
    # applied *within* a date only; nothing is carried forward across dates.
    df[names] = df[names].astype(float).fillna(0.0)

    valid_dates = df.assign(_v=df[names].notna().all(axis=1)).groupby("date")["_v"].sum()
    keep_dates = valid_dates[valid_dates >= min_names_per_date].index
    df = df[df["date"].isin(keep_dates)].reset_index(drop=True)

    log.info(
        f"AlphaDataset: {len(df):,} samples | {len(names)} features | "
        f"{df['date'].nunique()} dates | {df['symbol'].nunique()} symbols | horizon={horizon}"
    )
    return AlphaDataset(
        features=df[names],
        target=df["target"].astype(float),
        dates=df["date"],
        symbols=df["symbol"],
        forward_returns=df["fwd_return"].astype(float),
        feature_names=names,
        metadata={"horizon": horizon, "target_type": target},
    )


def to_matrix(dataset: AlphaDataset, mask: np.ndarray | None = None):
    """Return ``(X, y, meta)`` numpy/pandas views restricted to ``mask``."""
    if mask is None:
        X = dataset.features
        y = dataset.target
        meta = pd.DataFrame({"date": dataset.dates, "symbol": dataset.symbols})
    else:
        X = dataset.features[mask]
        y = dataset.target[mask]
        meta = pd.DataFrame({"date": dataset.dates[mask], "symbol": dataset.symbols[mask]})
    return X.to_numpy(dtype=float), y.to_numpy(dtype=float), meta


__all__ = ["AlphaDataset", "build_dataset", "to_matrix"]

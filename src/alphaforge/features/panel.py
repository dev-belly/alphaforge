"""Wide (dates x symbols) market panels derived from the long canonical table.

Everything downstream - factors, ML, optimiser, backtester - consumes the
:class:`MarketPanel` produced here, so shape and alignment are guaranteed in a
single place.

``returns`` is computed from **adjusted** prices. The forward-return helper is a
*label*, not a P&L: it measures ``t -> t + horizon`` from the signal date itself
unless an ``execution_lag`` is passed, so a fill model that can only trade at
``t + 1`` should ask for ``execution_lag=1`` - otherwise the label contains a
session of return the strategy never gets to hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("features.panel")


@dataclass
class MarketPanel:
    """Aligned wide panels plus the point-in-time investable universe."""

    dates: pd.DatetimeIndex
    close: pd.DataFrame  # adjusted close
    raw_close: pd.DataFrame
    returns: pd.DataFrame
    volume: pd.DataFrame
    dollar_volume: pd.DataFrame
    market_cap: pd.DataFrame
    universe: pd.DataFrame  # bool, point-in-time investable
    industry: pd.DataFrame  # (dates x symbols) industry label
    benchmark: pd.Series | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def symbols(self) -> pd.Index:
        return self.close.columns

    def __len__(self) -> int:
        return len(self.dates)

    def tradable(self) -> pd.DataFrame:
        """Close prices masked to the investable universe (NaN elsewhere)."""
        return self.close.where(self.universe)

    def forward_returns(self, horizon: int = 21, execution_lag: int = 0) -> pd.DataFrame:
        """``horizon``-period forward return, aligned to the *signal* date.

        Used as the **label** for factor evaluation and ML. Never feed this into
        a feature matrix.

        The window opens at ``t + execution_lag`` and closes at
        ``t + execution_lag + horizon``. It defaults to ``0`` (the window opens
        on the signal date), which is what every caller in this repo assumes; a
        fill model that can only trade the next session should pass ``1`` so the
        label does not include a session the strategy cannot capture.
        """
        entry = self.close.shift(-execution_lag)
        exit_ = self.close.shift(-(horizon + execution_lag))
        return exit_ / entry - 1.0

    def describe(self) -> dict:
        return {
            "dates": len(self.dates),
            "symbols": len(self.symbols),
            "start": str(self.dates.min().date()),
            "end": str(self.dates.max().date()),
            "avg_breadth": float(self.universe.sum(axis=1).mean()),
            "pct_obs": float(self.close.notna().values.mean()),
        }


def build_panel(
    prices: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    date_col: str = "date",
) -> MarketPanel:
    """Pivot the long price table into the canonical wide panels."""
    if prices.empty:
        raise ValueError("Cannot build a panel from an empty price frame")
    if "adj_close" not in prices.columns:
        # Without adjusted prices there is no return, no label and no universe:
        # an all-NaN panel would sail through as "zero breadth" instead.
        raise ValueError(
            "Cannot build a panel without an 'adj_close' column - prices, returns, "
            "the forward-return label and the universe are all derived from it"
        )

    df = prices.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, "symbol"])

    # Industry is metadata, not a price: a source that omits it - or leaves a
    # name unlabelled - gets the same explicit ``Unknown`` bucket the real-data
    # builder uses, so the dummy blocks and the neutralisation group means agree
    # on which names belong together.
    if "industry" in df.columns:
        df["industry"] = df["industry"].fillna("Unknown")
    else:
        df["industry"] = "Unknown"

    def pivot(value: str, how: str = "last") -> pd.DataFrame:
        if value not in df.columns:
            # Float, not object: an all-NaN *object* frame silently poisons every
            # downstream comparison and rolling window (``> 1e6`` raises,
            # ``.rolling(252).median()`` refuses non-numeric input).
            return pd.DataFrame(
                np.nan,
                index=sorted(df[date_col].unique()),
                columns=sorted(df["symbol"].unique()),
                dtype=float,
            )
        return df.pivot_table(index=date_col, columns="symbol", values=value, aggfunc=how)

    close = pivot("adj_close")
    raw_close = pivot("close")
    volume = pivot("volume")
    market_cap = pivot("market_cap")
    dollar_volume = close * volume

    # Some public sources (including the bundled real-data builder) carry no
    # shares-outstanding history, so a true market cap simply does not exist.
    # Rather than fabricate one - which would silently corrupt the size factor,
    # size neutralisation and every risk-model regression weight derived from it
    # - fall back to a *labelled* trailing dollar-volume proxy. The label rides
    # in ``metadata`` so reports can disclose the substitution instead of
    # presenting it as a real capitalisation.
    mcap_source = "reported"
    if market_cap.isna().all().all() or market_cap.empty:
        market_cap = dollar_volume.rolling(252, min_periods=60).median()
        if market_cap.isna().all().all():
            # No dollar volume either (or too little history) - say the size
            # factor is unavailable instead of advertising a proxy that is NaN.
            log.warning(
                "No market capitalisation and no usable dollar volume - market_cap "
                "is NaN, so the size factor and size neutralisation are empty."
            )
            mcap_source = "unavailable"
        else:
            log.warning(
                "No market capitalisation in the source data - using a trailing "
                "252d median dollar-volume proxy (metadata: market_cap_source)."
            )
            mcap_source = "dollar_volume_proxy"

    industry_long = (
        df[["symbol", "industry"]].drop_duplicates("symbol").set_index("symbol")["industry"]
    )
    industry = pd.DataFrame(
        np.tile(industry_long.reindex(close.columns).to_numpy(), (len(close), 1)),
        index=close.index,
        columns=close.columns,
    )

    if universe is None:
        universe = close.notna()
    else:
        universe = (
            universe.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        )

    # A name must have a price to be investable.
    universe &= close.notna()

    returns = close.pct_change(fill_method=None)

    panel = MarketPanel(
        dates=pd.DatetimeIndex(close.index),
        close=close,
        raw_close=raw_close,
        returns=returns,
        volume=volume,
        dollar_volume=dollar_volume,
        market_cap=market_cap,
        universe=universe,
        industry=industry,
        benchmark=benchmark,
        metadata={
            "provenance": str(getattr(prices, "attrs", {}).get("provenance", "UNKNOWN")),
            "market_cap_source": mcap_source,
        },
    )
    log.info(
        f"MarketPanel: {len(panel)} dates x {len(panel.symbols)} symbols | "
        f"avg breadth {universe.sum(axis=1).mean():.1f}"
    )
    return panel


def industry_dummies(industry: pd.DataFrame, drop_first: bool = True) -> dict[str, pd.DataFrame]:
    """One dummy panel per industry - the exposure block for neutralisation.

    Non-string labels (a missing industry, say) are dropped, so those names fall
    into the omitted category. With ``drop_first`` the omitted category is the
    alphabetically first one: the choice does not move the neutralised residual,
    but keeping it stable means adding a symbol cannot silently change which
    industry the intercept stands for.
    """
    labels = sorted(x for x in pd.unique(industry.values.ravel()) if isinstance(x, str))
    if drop_first and labels:
        labels = labels[1:]
    return {lab: industry.eq(lab).astype(float) for lab in labels}


__all__ = ["MarketPanel", "build_panel", "industry_dummies"]

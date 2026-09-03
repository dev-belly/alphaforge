"""Point-in-time universe construction.

Survivorship bias is the quietest way to inflate a backtest. This module keeps
the universe **as it actually was**: a name only enters on the date it was an
index member *and* passes the liquidity/price screens using information
available on that date. Names that later delist remain in the panel with their
terminal return - they are never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("data.universe")


@dataclass
class UniverseConfig:
    benchmark: str = "SP500_SAMPLE"
    min_price: float = 5.0
    min_adv_usd: float = 1_000_000.0
    adv_window: int = 21
    max_names: int | None = None
    # Require `min_history` observations before a name becomes investable so a
    # fresh listing cannot be traded on day one.
    min_history: int = 60


class Universe:
    """Builds and queries the investable set, strictly point-in-time."""

    def __init__(self, config: UniverseConfig | None = None) -> None:
        self.config = config or UniverseConfig()

    # ------------------------------------------------------------------
    def membership_matrix(
        self,
        constituents: pd.DataFrame,
        dates: pd.DatetimeIndex,
        symbols: pd.Index,
    ) -> pd.DataFrame:
        """Boolean (dates x symbols) membership from point-in-time snapshots.

        Membership is *forward filled from the snapshot date*, never backward:
        learning that a stock joined the index last month must not make it
        investable last month.
        """
        if constituents is None or constituents.empty:
            return pd.DataFrame(True, index=dates, columns=symbols)
        cons = constituents.copy()
        cons["date"] = pd.to_datetime(cons["date"])
        cons = cons.sort_values("date")
        matrix = pd.DataFrame(False, index=dates, columns=symbols)

        snapshots = cons["date"].drop_duplicates().sort_values().tolist()
        if not snapshots:
            return matrix
        # For each observation date, use the most recent snapshot at or before it.
        idx = np.searchsorted(pd.DatetimeIndex(snapshots), dates, side="right") - 1
        for i, d in enumerate(dates):
            k = idx[i]
            if k < 0:
                continue
            snap = snapshots[k]
            members = cons.loc[cons["date"] == snap, "symbol"]
            matrix.loc[d, matrix.columns.intersection(members)] = True
        return matrix

    # ------------------------------------------------------------------
    def eligibility_mask(
        self,
        prices: pd.DataFrame,
        dates: pd.DatetimeIndex,
        symbols: pd.Index,
    ) -> pd.DataFrame:
        """Liquidity / price / history screens, computed with trailing data only."""
        cfg = self.config
        panel = prices.pivot_table(
            index="date", columns="symbol", values="adj_close", aggfunc="last"
        )
        dollar_vol = panel * prices.pivot_table(
            index="date", columns="symbol", values="volume", aggfunc="last"
        )
        adv = dollar_vol.rolling(cfg.adv_window, min_periods=max(cfg.adv_window // 2, 5)).mean()
        history = panel.notna().rolling(cfg.min_history, min_periods=1).sum()

        panel = panel.reindex(index=dates, columns=symbols)
        adv = adv.reindex(index=dates, columns=symbols)
        history = history.reindex(index=dates, columns=symbols)

        mask = (
            panel.notna()
            & (panel >= cfg.min_price)
            & (adv.fillna(0.0) >= cfg.min_adv_usd)
            & (history.fillna(0) >= cfg.min_history)
        )
        return mask.fillna(False)

    # ------------------------------------------------------------------
    def build(
        self,
        prices: pd.DataFrame,
        constituents: pd.DataFrame,
        dates: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """Return the investable (dates x symbols) boolean panel."""
        if prices.empty:
            raise ValueError("Cannot build a universe from an empty price panel")
        panel = prices.pivot_table(
            index="date", columns="symbol", values="adj_close", aggfunc="last"
        )
        dates = pd.DatetimeIndex(dates) if dates is not None else panel.index
        symbols = panel.columns
        member = self.membership_matrix(constituents, dates, symbols)
        eligible = self.eligibility_mask(prices, dates, symbols)
        universe = member & eligible

        if self.config.max_names is not None:
            mcap = prices.pivot_table(
                index="date", columns="symbol", values="market_cap", aggfunc="last"
            ).reindex(index=dates, columns=symbols)
            keep = pd.DataFrame(False, index=dates, columns=symbols)
            for d in dates:
                row = universe.loc[d]
                if not row.any():
                    continue
                caps = mcap.loc[d, row].sort_values(ascending=False)
                if len(caps) > self.config.max_names:
                    row.loc[caps.index[self.config.max_names :]] = False
                keep.loc[d] = row
            universe = keep

        coverage = float(universe.values.mean())
        log.info(
            f"Universe built: {int(universe.sum().sum()):,} name-days | "
            f"avg breadth {universe.sum(axis=1).mean():.1f} | coverage {coverage:.1%}"
        )
        return universe

    # ------------------------------------------------------------------
    @staticmethod
    def survivorship_diagnostics(prices: pd.DataFrame, universe: pd.DataFrame) -> dict:
        """Quantify how much a survivorship-free universe differs from a naive one."""
        panel = prices.pivot_table(
            index="date", columns="symbol", values="adj_close", aggfunc="last"
        )
        all_syms = set(panel.columns)
        ever_investable = set(universe.columns[universe.any()])
        panel_end = panel.index.max()

        last_valid = panel.apply(lambda c: c.last_valid_index())
        delisted = [s for s, d in last_valid.items() if d is not None and (panel_end - d).days > 30]
        return {
            "n_symbols_total": len(all_syms),
            "n_symbols_investable": len(ever_investable),
            "n_delisted": len(delisted),
            "delisted_pct": len(delisted) / max(len(all_syms), 1),
            "note": (
                "Names that stop trading are retained with their terminal return. "
                "Filtering to currently-listed names would overstate performance."
            ),
        }


__all__ = ["Universe", "UniverseConfig"]

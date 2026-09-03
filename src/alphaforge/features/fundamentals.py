"""Point-in-time fundamental features.

The single most common look-ahead error in quant research is joining a
fundamental on its **fiscal period** instead of its **public release date**.
:class:`FundamentalView` makes that mistake impossible: every field is
materialised with ``merge_asof``-style logic keyed on ``report_date``, so on any
given trading day the model only sees statements that were already public.

An optional ``max_staleness_days`` guard prevents a five-year-old balance sheet
from lingering in the cross-section.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("features.fundamentals")

# Ratios derived from raw statement items; all expressed so that "higher = more
# of the factor", with the direction convention documented per name.
DERIVED_FIELDS = {
    # value
    "earnings_yield": ("net_income", "market_cap"),
    "book_to_price": ("total_equity", "market_cap"),
    "sales_to_price": ("revenue", "market_cap"),
    "fcf_yield": ("fcf", "market_cap"),
    "ebit_to_ev": ("ebit", "ev"),
    # quality
    "roe": ("net_income", "total_equity"),
    "roa": ("net_income", "total_assets"),
    "gross_profitability": ("gross_profit", "total_assets"),
    "asset_turnover": ("revenue", "total_assets"),
    "earnings_quality": ("ocf_minus_ni", "total_assets"),
    "accruals": ("ni_minus_ocf", "total_assets"),
    "gross_margin": ("gross_profit", "revenue"),
    "net_margin": ("net_income", "revenue"),
    # leverage / safety
    "leverage": ("total_debt", "total_assets"),
    "debt_to_equity": ("total_debt", "total_equity"),
}


@dataclass
class FundamentalView:
    """Point-in-time accessor over a canonical fundamentals table."""

    data: pd.DataFrame
    dates: pd.DatetimeIndex
    symbols: pd.Index
    market_cap: pd.DataFrame
    max_staleness_days: int = 550

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        fundamentals: pd.DataFrame,
        dates: pd.DatetimeIndex,
        symbols: pd.Index,
        market_cap: pd.DataFrame,
        max_staleness_days: int = 550,
    ) -> FundamentalView:
        if fundamentals is None or fundamentals.empty:
            log.warning("No fundamentals supplied - FundamentalView will be empty")
            return cls(
                data=pd.DataFrame(columns=["symbol", "report_date"]),
                dates=dates,
                symbols=symbols,
                market_cap=market_cap,
                max_staleness_days=max_staleness_days,
            )
        df = fundamentals.copy()
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df[df["report_date"].notna()]
        df = df.sort_values(["symbol", "report_date"]).reset_index(drop=True)

        # Derive the items needed for the ratio library.
        if {"operating_cashflow", "capex"} <= set(df.columns):
            df["fcf"] = df["operating_cashflow"] - df["capex"]
        if {"operating_cashflow", "net_income"} <= set(df.columns):
            df["ocf_minus_ni"] = df["operating_cashflow"] - df["net_income"]
            df["ni_minus_ocf"] = df["net_income"] - df["operating_cashflow"]
        if {"total_debt", "total_equity"} <= set(df.columns):
            df["ev"] = market_cap_estimate(df, market_cap) + df["total_debt"].fillna(0.0)

        return cls(
            data=df,
            dates=dates,
            symbols=symbols,
            market_cap=market_cap,
            max_staleness_days=max_staleness_days,
        )

    # ------------------------------------------------------------------
    def pit_panel(self, field: str) -> pd.DataFrame:
        """Wide (dates x symbols) panel of ``field`` as known on each date."""
        if field not in self.data.columns:
            return pd.DataFrame(np.nan, index=self.dates, columns=self.symbols)

        sub = self.data[["symbol", "report_date", field]].dropna(subset=[field])
        if sub.empty:
            return pd.DataFrame(np.nan, index=self.dates, columns=self.symbols)

        out = pd.DataFrame(np.nan, index=self.dates, columns=self.symbols, dtype=float)
        targets = self.dates.to_numpy("datetime64[ns]")
        max_gap = np.timedelta64(self.max_staleness_days, "D")

        for sym, grp in sub.groupby("symbol", sort=False):
            if sym not in out.columns:
                continue
            rd = grp["report_date"].to_numpy("datetime64[ns]")
            vals = grp[field].to_numpy(dtype=float)
            pos = np.searchsorted(rd, targets, side="right") - 1
            valid = pos >= 0
            picked = np.full(len(targets), np.nan)
            picked[valid] = vals[pos[valid]]
            # Staleness guard: drop observations whose release date is too old.
            rel = np.full(len(targets), np.datetime64("NaT"), dtype="datetime64[ns]")
            rel[valid] = rd[pos[valid]]
            too_old = (targets - rel) > max_gap
            picked[too_old] = np.nan
            out[sym] = picked
        return out

    # ------------------------------------------------------------------
    def ratio(self, name: str) -> pd.DataFrame:
        """Point-in-time ratio from :data:`DERIVED_FIELDS`."""
        if name not in DERIVED_FIELDS:
            raise KeyError(f"Unknown derived fundamental field: {name}")
        num_name, den_name = DERIVED_FIELDS[name]
        if den_name in {"market_cap", "ev"}:
            den = self.market_cap.reindex(index=self.dates, columns=self.symbols)
            if den_name == "ev":
                debt = self.pit_panel("total_debt")
                den = den + debt.fillna(0.0)
            num = self.pit_panel(num_name)
        else:
            num = self.pit_panel(num_name)
            den = self.pit_panel(den_name)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / den.replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def staleness(self) -> pd.DataFrame:
        """Days since the latest public release - a data-quality diagnostic."""
        sub = self.data[["symbol", "report_date"]].drop_duplicates()
        out = pd.DataFrame(np.nan, index=self.dates, columns=self.symbols)
        targets = self.dates.to_numpy("datetime64[ns]")
        for sym, grp in sub.groupby("symbol", sort=False):
            if sym not in out.columns:
                continue
            rd = np.sort(grp["report_date"].to_numpy("datetime64[ns]"))
            pos = np.searchsorted(rd, targets, side="right") - 1
            rel = np.full(len(targets), np.datetime64("NaT"), dtype="datetime64[ns]")
            ok = pos >= 0
            rel[ok] = rd[pos[ok]]
            out[sym] = (targets - rel) / np.timedelta64(1, "D")
        return out

    def available_fields(self) -> list[str]:
        return [c for c in self.data.columns if c not in {"symbol", "report_date", "fiscal_period"}]

    def coverage(self) -> pd.DataFrame:
        """Per-date fraction of the universe with any public fundamental."""
        any_field = self.pit_panel("total_assets") if "total_assets" in self.data.columns else None
        if any_field is None:
            return pd.DataFrame(index=self.dates, columns=["coverage"])
        return any_field.notna().mean(axis=1).to_frame("coverage")


def market_cap_estimate(df: pd.DataFrame, market_cap: pd.DataFrame) -> pd.Series:
    """Align a market-cap estimate onto a fundamentals frame's report dates."""
    mcap_long = (
        market_cap.stack()
        .rename("mcap")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "symbol"})
    )
    mcap_long["date"] = pd.to_datetime(mcap_long["date"])
    mcap_long = mcap_long.sort_values("date")
    target = df[["report_date", "symbol"]].copy()
    target = target.sort_values("report_date")
    merged = pd.merge_asof(
        target,
        mcap_long.rename(columns={"date": "report_date"}),
        on="report_date",
        by="symbol",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    )
    return merged["mcap"].fillna(np.nan).reset_index(drop=True)


__all__ = ["FundamentalView", "DERIVED_FIELDS"]

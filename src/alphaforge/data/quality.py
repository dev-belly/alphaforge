"""Data-quality diagnostics and cleaning primitives.

The single most expensive failure mode in quant research is silent bad data.
:class:`DataQualityReport` is run on every ETL pass and its findings are
persisted next to the artefacts so downstream users can see exactly what the
dataset does and does not support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("data.quality")

NUMERIC_PRICE_COLS = ["open", "high", "low", "close", "adj_close", "volume", "market_cap"]


@dataclass
class DataQualityReport:
    """Machine-readable quality summary for one ETL pass."""

    n_rows: int = 0
    n_symbols: int = 0
    n_dates: int = 0
    start_date: str = ""
    end_date: str = ""
    missing_pct: dict[str, float] = field(default_factory=dict)
    outlier_pct: dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0
    stale_observations: int = 0
    coverage: float = 0.0
    non_positive_prices: int = 0
    negative_volume: int = 0
    ohlc_violations: int = 0
    extreme_returns: int = 0
    delisting_candidates: int = 0
    survivorship_note: str = ""
    provenance: str = "UNKNOWN"
    warnings: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            ("rows", self.n_rows),
            ("symbols", self.n_symbols),
            ("dates", self.n_dates),
            ("coverage", round(self.coverage, 4)),
            ("duplicate_rows", self.duplicate_rows),
            ("stale_observations", self.stale_observations),
            ("non_positive_prices", self.non_positive_prices),
            ("negative_volume", self.negative_volume),
            ("ohlc_violations", self.ohlc_violations),
            ("extreme_returns", self.extreme_returns),
            ("delisting_candidates", self.delisting_candidates),
        ]
        rows += [(f"missing_pct[{k}]", round(v, 4)) for k, v in self.missing_pct.items()]
        rows += [(f"outlier_pct[{k}]", round(v, 4)) for k, v in self.outlier_pct.items()]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_symbols": self.n_symbols,
            "n_dates": self.n_dates,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "missing_pct": self.missing_pct,
            "outlier_pct": self.outlier_pct,
            "duplicate_rows": self.duplicate_rows,
            "stale_observations": self.stale_observations,
            "coverage": self.coverage,
            "non_positive_prices": self.non_positive_prices,
            "negative_volume": self.negative_volume,
            "ohlc_violations": self.ohlc_violations,
            "extreme_returns": self.extreme_returns,
            "delisting_candidates": self.delisting_candidates,
            "survivorship_note": self.survivorship_note,
            "provenance": self.provenance,
            "warnings": self.warnings,
        }

    def is_acceptable(self, max_missing: float = 0.35) -> bool:
        worst = max(self.missing_pct.get(c, 0.0) for c in ["close", "adj_close", "volume"])
        return worst <= max_missing and self.duplicate_rows == 0

    def summary(self) -> str:
        return (
            f"DataQualityReport(rows={self.n_rows:,}, symbols={self.n_symbols}, "
            f"dates={self.n_dates}, coverage={self.coverage:.1%}, "
            f"dups={self.duplicate_rows}, stale={self.stale_observations}, "
            f"provenance={self.provenance})"
        )


def build_quality_report(
    prices: pd.DataFrame,
    provenance: str = "UNKNOWN",
    stale_window: int = 5,
    extreme_return_threshold: float = 0.5,
) -> DataQualityReport:
    """Compute coverage / missingness / outlier statistics for a price panel."""
    if prices.empty:
        log.warning("Empty price panel - returning an empty quality report")
        return DataQualityReport(provenance=provenance, warnings=["empty panel"])

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    report = DataQualityReport(provenance=provenance)
    report.n_rows = len(df)
    report.n_symbols = int(df["symbol"].nunique())
    report.n_dates = int(df["date"].nunique())
    report.start_date = str(df["date"].min().date())
    report.end_date = str(df["date"].max().date())

    report.duplicate_rows = int(df.duplicated(subset=["date", "symbol"]).sum())
    report.coverage = float(len(df) / max(report.n_symbols * report.n_dates, 1))

    for col in NUMERIC_PRICE_COLS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        report.missing_pct[col] = float(series.isna().mean())
        clean = series.dropna()
        if len(clean) > 30:
            # Robust outlier detection on the log scale (prices are skewed).
            positive = clean[clean > 0]
            if len(positive) > 30:
                vals = np.log(positive)
                med = vals.median()
                mad = (vals - med).abs().median()
                if mad > 0:
                    z = 0.6745 * (vals - med) / mad
                    report.outlier_pct[col] = float((z.abs() > 5).mean())
                else:
                    report.outlier_pct[col] = 0.0

    close = pd.to_numeric(df.get("close"), errors="coerce")
    report.non_positive_prices = int((close <= 0).sum())
    vol = pd.to_numeric(df.get("volume"), errors="coerce")
    report.negative_volume = int((vol < 0).sum())

    if {"high", "low", "open", "close"} <= set(df.columns):
        h = pd.to_numeric(df["high"], errors="coerce")
        lo = pd.to_numeric(df["low"], errors="coerce")
        o = pd.to_numeric(df["open"], errors="coerce")
        c = pd.to_numeric(df["close"], errors="coerce")
        violation = (h < lo) | (h < c - 1e-9) | (h < o - 1e-9) | (lo > c + 1e-9) | (lo > o + 1e-9)
        report.ohlc_violations = int(violation.fillna(False).sum())

    # Stale observations: price repeated for `stale_window` consecutive days.
    panel = df.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last")
    if not panel.empty:
        unchanged = panel.diff().abs().fillna(1.0) < 1e-12
        runs = unchanged.rolling(stale_window).sum()
        report.stale_observations = int((runs >= stale_window).sum().sum())

        rets = panel.pct_change(fill_method=None)
        report.extreme_returns = int((rets.abs() > extreme_return_threshold).sum().sum())

        # A symbol whose last observation is well before the panel end is a
        # delisting / halt candidate -> the survivorship-bias surface.
        last_valid = panel.apply(lambda col: col.last_valid_index())
        end = panel.index.max()
        stale_syms = [
            sym for sym, lv in last_valid.items() if lv is not None and (end - lv).days > 30
        ]
        report.delisting_candidates = len(stale_syms)

    if report.duplicate_rows:
        report.warnings.append(f"{report.duplicate_rows} duplicate (date, symbol) rows detected")
    if report.non_positive_prices:
        report.warnings.append(f"{report.non_positive_prices} non-positive prices detected")
    if report.ohlc_violations:
        report.warnings.append(f"{report.ohlc_violations} OHLC consistency violations")
    if report.stale_observations:
        report.warnings.append(f"{report.stale_observations} stale price runs detected")
    if report.delisting_candidates:
        report.survivorship_note = (
            f"{report.delisting_candidates} symbols stop trading before the panel end and are "
            "retained in the dataset; dropping them would introduce survivorship bias."
        )
        report.warnings.append(report.survivorship_note)

    log.info(report.summary())
    return report


# --------------------------------------------------------------------------
# Cleaning primitives
# --------------------------------------------------------------------------
def clean_prices(
    prices: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_non_positive: bool = True,
    clip_extreme_returns: float | None = None,
) -> pd.DataFrame:
    """Apply the deterministic cleaning rules used by the ETL pipeline."""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    n0 = len(df)

    if drop_duplicates:
        df = df.drop_duplicates(subset=["date", "symbol"], keep="last")
    if drop_non_positive and "adj_close" in df.columns:
        df = df[pd.to_numeric(df["adj_close"], errors="coerce") > 0]
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        df = df[(vol.isna()) | (vol >= 0)]
    if clip_extreme_returns is not None:
        panel = df.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last")
        rets = panel.pct_change(fill_method=None)
        # Flag (do not silently overwrite) implausible moves for review.
        mask = rets.abs() > clip_extreme_returns
        n_flagged = int(mask.sum().sum())
        if n_flagged:
            log.warning(
                f"{n_flagged} return observations exceed {clip_extreme_returns:.0%} - flagged"
            )

    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    log.info(f"clean_prices: {n0:,} -> {len(df):,} rows")
    return df


def align_calendar(
    prices: pd.DataFrame,
    freq: str = "D",
    method: str = "ffill",
    limit: int = 5,
) -> pd.DataFrame:
    """Reindex the panel onto a regular calendar, forward-filling gaps.

    Forward-filling is capped at ``limit`` periods: carrying a stale price
    indefinitely is exactly how fake liquidity enters a backtest.
    """
    panel = prices.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last")
    full = pd.date_range(panel.index.min(), panel.index.max(), freq=freq)
    aligned = panel.reindex(full)
    if method == "ffill":
        aligned = aligned.ffill(limit=limit)
    elif method == "bfill":
        aligned = aligned.bfill(limit=limit)
    aligned.index.name = "date"
    return aligned


def detect_missing_blocks(panel: pd.DataFrame, min_length: int = 5) -> pd.DataFrame:
    """Return a tidy frame of contiguous NaN blocks longer than ``min_length``."""
    rows = []
    for sym in panel.columns:
        col = panel[sym]
        isna = col.isna()
        if not isna.any():
            continue
        grp = (isna != isna.shift()).cumsum()
        for _, block in isna.groupby(grp):
            if block.all() and len(block) >= min_length:
                rows.append(
                    {
                        "symbol": sym,
                        "start": block.index[0],
                        "end": block.index[-1],
                        "length": len(block),
                    }
                )
    return pd.DataFrame(rows, columns=["symbol", "start", "end", "length"])


__all__ = [
    "DataQualityReport",
    "build_quality_report",
    "clean_prices",
    "align_calendar",
    "detect_missing_blocks",
]

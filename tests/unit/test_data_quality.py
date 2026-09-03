"""Unit tests for the data-quality layer (report + cleaning + calendar align)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.quality import (
    align_calendar,
    build_quality_report,
    clean_prices,
    detect_missing_blocks,
)


def _long_prices(n_dates: int = 120, n_syms: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    syms = [f"S{i}" for i in range(n_syms)]
    rows = []
    base = rng.uniform(50, 200, size=n_syms)
    for d in idx:
        for j, s in enumerate(syms):
            px = base[j] * (1.0 + rng.normal(0, 0.01))
            rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "open": px * 0.99,
                    "high": px * 1.02,
                    "low": px * 0.98,
                    "close": px,
                    "adj_close": px,
                    "volume": rng.uniform(1e5, 1e6),
                    "market_cap": px * 1e7,
                    "industry": "X",
                }
            )
    return pd.DataFrame(rows)


def test_quality_report_basic_stats():
    df = _long_prices()
    rep = build_quality_report(df, provenance="sample")
    assert rep.n_symbols == 6
    assert rep.n_dates == 120
    assert 0.0 <= rep.coverage <= 1.0
    assert rep.duplicate_rows == 0
    assert rep.is_acceptable()


def test_quality_report_flags_duplicates():
    df = _long_prices()
    dup = df.iloc[[0]].copy()
    rep = build_quality_report(pd.concat([df, dup], ignore_index=True), provenance="sample")
    assert rep.duplicate_rows >= 1
    assert not rep.is_acceptable()


def test_quality_report_detects_non_positive_price():
    df = _long_prices()
    df.loc[df.index[0], "adj_close"] = 0.0
    df.loc[df.index[0], "close"] = 0.0
    rep = build_quality_report(df, provenance="sample")
    assert rep.non_positive_prices >= 1
    assert any("non-positive" in w for w in rep.warnings)


def test_clean_prices_drops_duplicates_and_nonpositive():
    df = _long_prices()
    dup = df.iloc[[0]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    df.loc[df.index[1], "adj_close"] = -1.0
    cleaned = clean_prices(df, drop_non_positive=True)
    assert len(cleaned) == len(df) - 2  # removed the dup + the bad price


def test_align_calendar_ffill_is_capped():
    df = _long_prices(n_dates=60)
    # Punch a 10-session hole in one symbol by dropping those rows.
    drop_dates = sorted(df["date"].unique())[10:20]
    df = df[~((df["symbol"] == "S0") & df["date"].isin(drop_dates))]
    aligned = align_calendar(df, freq="B", method="ffill", limit=5)
    # A hole longer than the ffill limit must remain NaN somewhere within it.
    hole = aligned["S0"].reindex(drop_dates)
    assert hole.isna().any()


def test_detect_missing_blocks_finds_long_gap():
    df = _long_prices(n_dates=60)
    # Drop a contiguous 12-session block for one symbol.
    drop_dates = sorted(df["date"].unique())[10:22]
    df = df[~((df["symbol"] == "S1") & df["date"].isin(drop_dates))]
    panel = df.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last")
    blocks = detect_missing_blocks(panel, min_length=5)
    assert not blocks.empty
    assert (blocks["symbol"] == "S1").any()
    assert (blocks["length"] >= 10).any()

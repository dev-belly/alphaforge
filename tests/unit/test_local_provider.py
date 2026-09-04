"""Offline tests for the local Parquet provider.

``LocalParquetProvider`` is the default production backend and the one that
makes research reproducible from committed artefacts, so its filtering and
schema-validation behaviour is worth locking down. Everything here runs against
a ``tmp_path`` Parquet store - no vendor, no network, no DuckDB server.

The behaviour that matters most is *point-in-time discipline*: fundamentals are
filtered on ``report_date`` (when the market could see the number), never on
``fiscal_period`` (when it was earned).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.data.providers.base import (
    CONSTITUENT_COLUMNS,
    MACRO_COLUMNS,
    PRICE_COLUMNS,
)
from alphaforge.data.providers.local import LocalParquetProvider


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    return tmp_path


def _write(store: Path, name: str, df: pd.DataFrame) -> None:
    df.to_parquet(store / name, index=False)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-02", "2024-01-03"]
            ),
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "open": [10.0, 11.0, 12.0, 20.0, 21.0],
            "high": [10.5, 11.5, 12.5, 20.5, 21.5],
            "low": [9.5, 10.5, 11.5, 19.5, 20.5],
            "close": [10.2, 11.2, 12.2, 20.2, 21.2],
            "adj_close": [10.1, 11.1, 12.1, 20.1, 21.1],
            "volume": [100.0, 110.0, 120.0, 200.0, 210.0],
            "market_cap": [np.nan] * 5,
            "shares_outstanding": [np.nan] * 5,
            "industry": ["Unknown"] * 5,
        }
    )


# --------------------------------------------------------------------------
# Schema handling
# --------------------------------------------------------------------------
def test_missing_table_returns_empty_canonical_frame(store: Path) -> None:
    """Absent tables must degrade to an empty frame with the right schema."""
    provider = LocalParquetProvider(store)
    prices = provider.fetch_prices(["AAA"], "2024-01-01", "2024-01-31")
    assert list(prices.columns) == PRICE_COLUMNS and prices.empty

    macro = provider.fetch_macro(["VIX"], "2024-01-01", "2024-01-31")
    assert list(macro.columns) == MACRO_COLUMNS and macro.empty

    const = provider.fetch_constituents("SP500", "2024-01-01", "2024-01-31")
    assert list(const.columns) == CONSTITUENT_COLUMNS and const.empty


def test_table_missing_required_columns_raises(store: Path) -> None:
    """A schema mismatch fails loudly instead of silently dropping columns."""
    _write(store, "prices.parquet", pd.DataFrame({"symbol": ["AAA"], "close": [1.0]}))
    provider = LocalParquetProvider(store)
    with pytest.raises(ValueError, match="missing required columns"):
        provider.fetch_prices(["AAA"], "2024-01-01", "2024-01-31")


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def test_fetch_prices_filters_symbols_and_dates(store: Path) -> None:
    _write(store, "prices.parquet", _prices())
    provider = LocalParquetProvider(store)

    out = provider.fetch_prices(["AAA"], "2024-01-03", "2024-01-31")
    assert set(out["symbol"]) == {"AAA"}
    assert out["date"].min() == pd.Timestamp("2024-01-03")
    assert len(out) == 2

    both = provider.fetch_prices(["AAA", "BBB"], "2024-01-01", "2024-01-31")
    assert len(both) == 5
    assert both["date"].is_monotonic_increasing  # always sorted by date
    assert list(both.index) == list(range(len(both)))  # index reset


def test_fetch_prices_accepts_path_or_string_root(tmp_path: Path) -> None:
    _write(tmp_path, "prices.parquet", _prices())
    from_str = LocalParquetProvider(str(tmp_path)).fetch_prices(["AAA"], "2024-01-01", "2024-01-31")
    from_path = LocalParquetProvider(tmp_path).fetch_prices(["AAA"], "2024-01-01", "2024-01-31")
    pd.testing.assert_frame_equal(from_str, from_path)


def test_fetch_fundamentals_filters_on_report_date(store: Path) -> None:
    """Point-in-time: the filter key is report_date, not fiscal_period."""
    _write(
        store,
        "fundamentals.parquet",
        pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "fiscal_period": ["2023Q4", "2024Q1"],
                "report_date": pd.to_datetime(["2024-02-01", "2024-05-01"]),
                "revenue": [100.0, 120.0],
            }
        ),
    )
    provider = LocalParquetProvider(store)
    out = provider.fetch_fundamentals(["AAA"], "2024-03-01", "2024-12-31")
    assert len(out) == 1
    assert out["fiscal_period"].iloc[0] == "2024Q1"


def test_fetch_constituents_filters_index_id(store: Path) -> None:
    _write(
        store,
        "constituents.parquet",
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
                "symbol": ["AAA", "BBB"],
                "index_id": ["SP500", "CSI300"],
                "weight": [0.5, 0.5],
            }
        ),
    )
    provider = LocalParquetProvider(store)
    out = provider.fetch_constituents("SP500", "2024-01-01", "2024-12-31")
    assert out["symbol"].tolist() == ["AAA"]


def test_fetch_macro_filters_series_ids(store: Path) -> None:
    _write(
        store,
        "macro.parquet",
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "series_id": ["VIX", "TNX"],
                "value": [17.5, 4.2],
            }
        ),
    )
    provider = LocalParquetProvider(store)
    out = provider.fetch_macro(["VIX"], "2024-01-01", "2024-01-31")
    assert out["series_id"].tolist() == ["VIX"]


def test_fetch_industry_filters_symbols(store: Path) -> None:
    _write(
        store,
        "industry.parquet",
        pd.DataFrame({"symbol": ["AAA", "BBB"], "industry": ["Tech", "Energy"]}),
    )
    provider = LocalParquetProvider(store)
    out = provider.fetch_industry(["BBB"])
    assert out["industry"].tolist() == ["Energy"]
    assert list(out.columns) == ["symbol", "industry"]


# --------------------------------------------------------------------------
# benchmark_prices()
# --------------------------------------------------------------------------
def test_benchmark_prices_is_sorted_and_date_filtered(store: Path) -> None:
    # Deliberately written out of chronological order.
    _write(
        store,
        "benchmark.parquet",
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-05"]),
                "value": [101.0, 100.0, 103.0],
            }
        ),
    )
    provider = LocalParquetProvider(store)
    bench = provider.benchmark_prices("SP500", "2024-01-02", "2024-01-03")

    assert bench.name == "benchmark"
    assert bench.index.is_monotonic_increasing  # sorted regardless of file order
    assert bench.tolist() == [100.0, 101.0]


def test_benchmark_prices_raises_when_absent(store: Path) -> None:
    provider = LocalParquetProvider(store)
    with pytest.raises(FileNotFoundError, match="No benchmark series stored locally"):
        provider.benchmark_prices("SP500", "2024-01-01", "2024-01-31")

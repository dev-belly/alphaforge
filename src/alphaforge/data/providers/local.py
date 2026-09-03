"""Local Parquet / DuckDB provider - the default production backend.

Reads the canonical long-format tables written by :class:`~alphaforge.data.storage.DataStore`
so that research runs are reproducible from committed artefacts without hitting
any vendor API.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from alphaforge.data.providers.base import (
    CONSTITUENT_COLUMNS,
    MACRO_COLUMNS,
    PRICE_COLUMNS,
    DataProvider,
)
from alphaforge.utils.logging import get_logger

log = get_logger("data.local")


class LocalParquetProvider(DataProvider):
    """Serves canonical tables from ``data/processed`` (or any given root)."""

    name = "local"

    def __init__(self, root: str | Path = "data/processed") -> None:
        self.root = Path(root)

    def _read(self, filename: str, required: Sequence[str]) -> pd.DataFrame:
        path = self.root / filename
        if not path.exists():
            log.warning(f"Local table missing: {path}")
            return pd.DataFrame({c: pd.Series(dtype="object") for c in required})
        df = pd.read_parquet(path)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        return df

    def fetch_prices(
        self, symbols: Sequence[str] | None = None, start=None, end=None
    ) -> pd.DataFrame:
        df = self._read("prices.parquet", PRICE_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        return self._filter(df, symbols, start, end)

    def fetch_fundamentals(
        self, symbols: Sequence[str] | None = None, start=None, end=None
    ) -> pd.DataFrame:
        df = self._read("fundamentals.parquet", ["symbol", "fiscal_period", "report_date"])
        df["report_date"] = pd.to_datetime(df["report_date"])
        return self._filter(df, symbols, start, end, date_col="report_date")

    def fetch_constituents(self, index_id: str | None = None, start=None, end=None) -> pd.DataFrame:
        df = self._read("constituents.parquet", CONSTITUENT_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        if index_id is not None and "index_id" in df.columns and not df.empty:
            df = df[df["index_id"] == index_id]
        return self._filter(df, None, start, end)

    def fetch_macro(
        self, series: Sequence[str] | None = None, start=None, end=None
    ) -> pd.DataFrame:
        df = self._read("macro.parquet", MACRO_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        if series and not df.empty:
            df = df[df["series_id"].isin(list(series))]
        return self._filter(df, None, start, end)

    def fetch_industry(self, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        df = self._read("industry.parquet", ["symbol", "industry"])
        if symbols is not None and not df.empty:
            df = df[df["symbol"].isin(list(symbols))]
        return df

    def benchmark_prices(self, index_id: str | None = None, start=None, end=None) -> pd.Series:
        df = self._read("benchmark.parquet", ["date", "value"])
        if df.empty:
            raise FileNotFoundError("No benchmark series stored locally")
        s = pd.Series(df["value"].to_numpy(), index=pd.to_datetime(df["date"]), name="benchmark")
        s = s.sort_index()
        if start:
            s = s[s.index >= pd.Timestamp(start)]
        if end:
            s = s[s.index <= pd.Timestamp(end)]
        return s

    @staticmethod
    def _filter(
        df: pd.DataFrame,
        symbols: Sequence[str] | None,
        start,
        end,
        date_col: str = "date",
    ) -> pd.DataFrame:
        if df.empty:
            return df
        if symbols is not None and "symbol" in df.columns:
            df = df[df["symbol"].isin(list(symbols))]
        if start is not None:
            df = df[df[date_col] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df[date_col] <= pd.Timestamp(end)]
        return df.sort_values([date_col]).reset_index(drop=True)


__all__ = ["LocalParquetProvider"]

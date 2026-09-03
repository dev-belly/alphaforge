"""Canonical schemas and the abstract :class:`DataProvider` contract.

AlphaForge never binds the research engine to a single vendor. Every source
(Yahoo Finance, AkShare, Tushare, local Parquet, bundled sample data) is
exposed through the same four-method interface so that swapping providers is a
configuration change, not a refactor.

Canonical long-format schema
----------------------------
prices        : date, symbol, open, high, low, close, adj_close, volume,
                market_cap, turnover, shares_outstanding, industry
fundamentals : symbol, fiscal_period, report_date (public release),
                revenue, net_income, total_assets, total_equity,
                operating_cashflow, capex, total_debt, gross_profit, ebit
macro        : date, series_id, value
constituents : date, symbol, index_id, weight
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

PRICE_COLUMNS: list[str] = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "market_cap",
    "shares_outstanding",
    "industry",
]

FUNDAMENTAL_COLUMNS: list[str] = [
    "symbol",
    "fiscal_period",
    "report_date",
    "revenue",
    "cogs",
    "gross_profit",
    "ebit",
    "net_income",
    "total_assets",
    "total_equity",
    "total_debt",
    "operating_cashflow",
    "capex",
]

MACRO_COLUMNS: list[str] = ["date", "series_id", "value"]

CONSTITUENT_COLUMNS: list[str] = ["date", "symbol", "index_id", "weight"]

# Industries follow the 11 GICS sector buckets.
GICS_SECTORS: list[str] = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]


@dataclass
class DataBundle:
    """Container returned by an ETL run - everything downstream consumes."""

    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    macro: pd.DataFrame
    constituents: pd.DataFrame
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class DataProvider(ABC):
    """Unified market-data interface implemented by every adapter."""

    name: str = "base"

    @abstractmethod
    def fetch_prices(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        """Return long-format OHLCV + adjusted price panel."""

    @abstractmethod
    def fetch_fundamentals(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        """Return fundamentals keyed by *public release* date (``report_date``)."""

    @abstractmethod
    def fetch_constituents(self, index_id: str, start: str, end: str) -> pd.DataFrame:
        """Return historical index membership / weights."""

    @abstractmethod
    def fetch_macro(self, series: Sequence[str], start: str, end: str) -> pd.DataFrame:
        """Return macro time series in long format."""

    @abstractmethod
    def fetch_industry(self, symbols: Sequence[str]) -> pd.DataFrame:
        """Return ``symbol -> industry`` classification map."""

    # -- shared helpers ---------------------------------------------------
    @staticmethod
    def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

    def benchmark_prices(self, index_id: str, start: str, end: str) -> pd.Series:
        """Convenience: total-return benchmark series used for relative analytics."""
        raise NotImplementedError(f"{self.name} does not provide benchmark series")

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "capabilities": {
                "prices": True,
                "fundamentals": True,
                "constituents": True,
                "macro": True,
            },
        }

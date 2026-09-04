"""Vendor adapters that require network access or third-party SDKs.

Both adapters degrade gracefully: if the optional dependency is not installed
the constructor raises :class:`ProviderUnavailableError` with an actionable
message and the platform falls back to the local/sample provider.

Neither adapter stores credentials - any token must come from the environment
(never committed) and is read at call time.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphaforge.data.providers.base import (
    CONSTITUENT_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    MACRO_COLUMNS,
    PRICE_COLUMNS,
    DataProvider,
)
from alphaforge.utils.logging import get_logger

log = get_logger("data.vendors")

# Curated, always-liquid universes used when callers do not supply an explicit
# symbol list. These are fallbacks so a real backtest can run with zero config:
# the pipeline resolves them automatically for the live vendors.
YAHOO_DEFAULT_UNIVERSE: list[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK-B",
    "JPM",
    "V",
    "UNH",
    "JNJ",
    "WMT",
    "PG",
    "MA",
    "HD",
    "BAC",
    "XOM",
    "KO",
    "PEP",
    "COST",
    "CVX",
    "ABBV",
    "TMO",
    "AVGO",
    "ORCL",
    "ADBE",
    "MCD",
    "CRM",
    "AMD",
]

AKSHARE_DEFAULT_UNIVERSE: list[str] = [
    "600519",
    "601318",
    "600036",
    "000858",
    "601166",
    "600276",
    "000333",
    "002594",
    "601012",
    "600900",
    "000651",
    "600030",
    "601888",
    "600887",
    "000001",
    "601398",
    "600585",
    "002415",
    "600309",
    "601899",
    "000725",
    "002475",
    "601988",
    "600028",
    "601857",
    "600104",
    "000002",
    "002714",
    "600809",
    "603259",
    "688981",
    "300750",
    "300059",
    "002230",
    "600690",
    "601668",
    "600048",
    "000063",
    "002241",
    "600760",
]


class ProviderUnavailableError(RuntimeError):
    """Raised when an optional data vendor cannot be used in this environment."""


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance adapter via ``yfinance`` (optional dependency).

    Notes / limitations (documented in README):
      * Yahoo provides *current* index membership only -> historical
        constituents are unavailable, so survivorship bias cannot be removed.
      * Fundamental release timestamps are not exposed; the adapter therefore
        applies ``fundamental_lag_days`` from config as a conservative
        point-in-time approximation.
    """

    name = "yahoo"

    def __init__(self, fundamental_lag_days: int = 90, cache_dir: str | None = None) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ProviderUnavailableError(
                "yfinance is not installed. Install it with `pip install yfinance`."
            ) from exc
        import yfinance as yf

        self._yf = yf
        self.fundamental_lag_days = fundamental_lag_days
        self.cache_dir = cache_dir

    def fetch_prices(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        tickers = list(symbols)
        log.info(f"Yahoo: downloading {len(tickers)} tickers [{start} -> {end}]")
        raw = self._yf.download(
            tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker"
        )
        frames = []
        for sym in tickers:
            if isinstance(raw.columns, pd.MultiIndex):
                if sym not in raw.columns.get_level_values(0):
                    continue
                df = raw[sym].copy()
            else:
                if len(tickers) != 1:
                    continue
                df = raw.copy()
            df = df.reset_index().rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            df["symbol"] = sym
            # Best-effort market cap from the fast info endpoint. When it is
            # unavailable the risk model gracefully falls back to equal weights.
            mcap = np.nan
            try:
                shares = self._yf.Ticker(sym).fast_info.shares_outstanding
                if shares is not None and not pd.isna(shares):
                    mcap = float(shares) * float(df["close"].iloc[-1])
            except Exception as exc:  # noqa: BLE001
                log.debug(f"Yahoo market cap unavailable for {sym}: {exc}")
            df["market_cap"] = mcap
            df["shares_outstanding"] = np.nan
            df["industry"] = "Unknown"
            frames.append(df)
        if not frames:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in PRICE_COLUMNS})
        out = pd.concat(frames, ignore_index=True)
        return out[[c for c in PRICE_COLUMNS if c in out.columns]]

    def fetch_fundamentals(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        log.warning(
            "Yahoo exposes no point-in-time release dates; applying a "
            f"{self.fundamental_lag_days}-day conservative publication lag."
        )
        rows = []
        for sym in symbols:
            try:
                tk = self._yf.Ticker(sym)
                bs = tk.quarterly_balance_sheet
                inc = tk.quarterly_financials
                cf = tk.quarterly_cashflow
                if bs is None or bs.empty:
                    continue
                for col in bs.columns:
                    period_end = pd.Timestamp(col).normalize()

                    def pick(frame, names, col=col):
                        if frame is None or frame.empty:
                            return np.nan
                        for n in names:
                            if n in frame.index:
                                return float(frame.loc[n, col])
                        return np.nan

                    rows.append(
                        {
                            "symbol": sym,
                            "fiscal_period": str(pd.Period(period_end, freq="Q")),
                            "period_end": period_end,
                            "report_date": period_end
                            + pd.Timedelta(days=self.fundamental_lag_days),
                            "revenue": pick(inc, ["Total Revenue", "Revenue"]),
                            "cogs": pick(inc, ["Cost Of Revenue", "Cost of Revenue"]),
                            "gross_profit": pick(inc, ["Gross Profit"]),
                            "ebit": pick(inc, ["EBIT", "Operating Income"]),
                            "net_income": pick(inc, ["Net Income"]),
                            "total_assets": pick(bs, ["Total Assets"]),
                            "total_equity": pick(
                                bs, ["Stockholders Equity", "Total Stockholder Equity"]
                            ),
                            "total_debt": pick(bs, ["Total Debt", "Long Term Debt"]),
                            "operating_cashflow": pick(
                                cf, ["Operating Cash Flow", "Total Cash From Operating Activities"]
                            ),
                            "capex": pick(cf, ["Capital Expenditures"]),
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - vendor calls are best effort
                log.warning(f"Yahoo fundamentals failed for {sym}: {exc}")
        if not rows:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in FUNDAMENTAL_COLUMNS})
        df = pd.DataFrame(rows)
        return df[[c for c in FUNDAMENTAL_COLUMNS if c in df.columns] + ["period_end"]]

    def fetch_constituents(self, index_id: str, start: str, end: str) -> pd.DataFrame:
        """Yahoo only exposes *current* membership - survivorship bias applies."""
        log.warning(
            f"Yahoo cannot supply historical membership for {index_id}; "
            "returning current snapshot with equal weights (survivorship bias!)."
        )
        try:
            table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
            syms = table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        except Exception as exc:  # noqa: BLE001
            log.error(f"Failed to scrape constituent list: {exc}")
            return pd.DataFrame({c: pd.Series(dtype="object") for c in CONSTITUENT_COLUMNS})
        dates = pd.bdate_range(start, end, freq="ME")
        rows = [
            {"date": d, "symbol": s, "index_id": index_id, "weight": 1.0 / len(syms)}
            for d in dates
            for s in syms
        ]
        return pd.DataFrame(rows)[CONSTITUENT_COLUMNS]

    def fetch_macro(self, series: Sequence[str], start: str, end: str) -> pd.DataFrame:
        tickers = {"VIX": "^VIX", "TNX": "^TNX", "USD_INDEX": "DX-Y.NYB"}
        rows = []
        for sid in series:
            tk = tickers.get(sid)
            if tk is None:
                continue
            try:
                s = self._yf.download(tk, start=start, end=end, progress=False)["Adj Close"]
                for d, v in s.items():
                    rows.append({"date": d, "series_id": sid, "value": float(v)})
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Yahoo macro fetch failed for {sid}: {exc}")
        if not rows:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in MACRO_COLUMNS})
        return pd.DataFrame(rows)[MACRO_COLUMNS]

    def fetch_industry(self, symbols: Sequence[str]) -> pd.DataFrame:
        rows = []
        for sym in symbols:
            try:
                info = self._yf.Ticker(sym).info or {}
                rows.append(
                    {
                        "symbol": sym,
                        "industry": info.get("sector") or info.get("industry") or "Unknown",
                    }
                )
            except Exception:  # noqa: BLE001
                rows.append({"symbol": sym, "industry": "Unknown"})
        return pd.DataFrame(rows, columns=["symbol", "industry"])

    def benchmark_prices(self, index_id: str = "^GSPC", start=None, end=None) -> pd.Series:
        s = self._yf.download(index_id, start=start, end=end, progress=False)["Adj Close"]
        s.name = "benchmark"
        return s


class EastMoneyProvider(DataProvider):
    """Key-less A-share data provider backed by the EastMoney quote API.

    This is the same data source AkShare's ``stock_zh_a_hist`` uses, but we call
    the HTTP endpoint directly so the platform needs **no third-party SDK** and
    no API token. It is therefore the default ``akshare`` backend in AlphaForge:
    the ``akshare`` package is heavy and its schema changes without notice.

    Limitations (documented in the README):
      * Only daily OHLCV + a qfq-adjusted close are fetched. ``market_cap`` is
        left NaN, so the risk model falls back to equal-weight, and
        value/quality factors are unavailable for this run (no fundamentals).
      * EastMoney exposes current index membership only -> survivorship bias
        applies; constituents are returned empty and the universe falls back to
        the liquidity / price screens.
    """

    name = "eastmoney"
    _KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    # EastMoney's public ``ut`` endpoint parameter (a non-credential, public API
    # constant also embedded in AkShare). Override via the ALPHAFORGE_EASTMONEY_UT
    # environment variable; no private key is ever stored in the repository.
    _UT = os.environ.get("ALPHAFORGE_EASTMONEY_UT", "fa5fd1943c7b386f172d6893dbfba10b")

    def __init__(self, timeout: int = 20) -> None:
        try:
            import requests
        except ImportError as exc:
            raise ProviderUnavailableError(
                "requests is not installed. Install it with `pip install requests`."
            ) from exc

        self._requests = requests
        self.timeout = timeout
        log.info("EastMoney provider initialised (A-share market, key-less)")

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _secid(code: str) -> str:
        """Map a 6-digit A-share code to EastMoney's ``market.code`` secid."""
        code = str(code).zfill(6)
        if code.startswith("6"):
            return f"1.{code}"  # Shanghai
        if code.startswith(("0", "3", "8", "4")):
            return f"0.{code}"  # Shenzhen / Beijing
        return f"1.{code}"

    @staticmethod
    def _index_secid(index_id: str) -> str:
        """Map an index code (e.g. 000300) to an EastMoney secid."""
        if "." in str(index_id):
            return str(index_id)
        code = str(index_id).zfill(6)
        if code.startswith("399"):
            return f"0.{code}"  # Shenzhen indices (e.g. 399001, 399006)
        return f"1.{code}"  # Shanghai indices (incl. CSI 300 = 000300)

    def _klines(self, secid: str, start: str, end: str) -> list[str]:
        params = {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56",  # date,open,close,high,low,volume
            "klt": "101",  # daily
            "fqt": "1",  # qfq (forward-adjusted)
            "beg": str(start).replace("-", ""),
            "end": str(end).replace("-", ""),
            "ut": self._UT,
        }
        resp = self._requests.get(self._KLINE_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json().get("data")
        if not data or not data.get("klines"):
            return []
        return list(data["klines"])

    # -- interface ------------------------------------------------------
    def fetch_prices(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        frames = []
        for sym in symbols:
            try:
                klines = self._klines(self._secid(sym), start, end)
                if not klines:
                    log.warning(f"EastMoney: no klines for {sym}")
                    continue
                recs = []
                for row in klines:
                    p = row.split(",")
                    if len(p) < 6:
                        continue
                    recs.append(
                        {
                            "date": pd.Timestamp(p[0]),
                            "symbol": str(sym),
                            "open": float(p[1]),
                            "close": float(p[2]),
                            "high": float(p[3]),
                            "low": float(p[4]),
                            "volume": float(p[5]),
                        }
                    )
                if not recs:
                    continue
                df = pd.DataFrame(recs)
                df["adj_close"] = df["close"]  # qfq-adjusted close
                df["market_cap"] = np.nan
                df["shares_outstanding"] = np.nan
                df["industry"] = "Unknown"
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"EastMoney price fetch failed for {sym}: {exc}")
        if not frames:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in PRICE_COLUMNS})
        out = pd.concat(frames, ignore_index=True)
        return out[[c for c in PRICE_COLUMNS if c in out.columns]]

    def fetch_fundamentals(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        log.warning(
            "EastMoney provider supplies price data only; value/quality factors "
            "are unavailable for this run (fundamentals not fetched)."
        )
        return pd.DataFrame({c: pd.Series(dtype="object") for c in FUNDAMENTAL_COLUMNS})

    def fetch_constituents(self, index_id: str, start: str, end: str) -> pd.DataFrame:
        log.warning(
            f"EastMoney does not supply historical membership for {index_id}; "
            "returning empty (universe falls back to liquidity screens)."
        )
        return pd.DataFrame({c: pd.Series(dtype="object") for c in CONSTITUENT_COLUMNS})

    def fetch_macro(self, series: Sequence[str], start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in MACRO_COLUMNS})

    def fetch_industry(self, symbols: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [{"symbol": str(s), "industry": "Unknown"} for s in symbols],
            columns=["symbol", "industry"],
        )

    def benchmark_prices(self, index_id: str = "000300", start=None, end=None) -> pd.Series:
        secid = self._index_secid(index_id)
        klines = self._klines(secid, start or "2000-01-01", end or "2030-01-01")
        if not klines:
            raise RuntimeError(f"EastMoney benchmark unavailable for {index_id}")
        dates, closes = [], []
        for row in klines:
            p = row.split(",")
            if len(p) < 3:
                continue
            dates.append(pd.Timestamp(p[0]))
            closes.append(float(p[2]))
        return pd.Series(closes, index=pd.DatetimeIndex(dates), name="benchmark")


__all__ = [
    "YahooFinanceProvider",
    "EastMoneyProvider",
    "ProviderUnavailableError",
]

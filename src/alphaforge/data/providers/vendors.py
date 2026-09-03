"""Vendor adapters that require network access or third-party SDKs.

Both adapters degrade gracefully: if the optional dependency is not installed
the constructor raises :class:`ProviderUnavailableError` with an actionable
message and the platform falls back to the local/sample provider.

Neither adapter stores credentials - any token must come from the environment
(never committed) and is read at call time.
"""

from __future__ import annotations

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
            df["market_cap"] = np.nan
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


class AkShareProvider(DataProvider):
    """AkShare adapter for A-share data (optional dependency).

    AkShare is community-maintained; schemas change without notice, so every
    call is defensive and normalises into the AlphaForge canonical schema.
    """

    name = "akshare"

    def __init__(self, token: str | None = None) -> None:
        try:
            import akshare  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ProviderUnavailableError(
                "akshare is not installed. Install it with `pip install akshare`."
            ) from exc
        import akshare as ak

        self._ak = ak
        # Tokens (e.g. Tushare) are read from the environment only.
        self.token = token
        log.info("AkShare provider initialised (A-share market)")

    def fetch_prices(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        frames = []
        s_start, s_end = start.replace("-", ""), end.replace("-", "")
        for sym in symbols:
            try:
                df = self._ak.stock_zh_a_hist(
                    symbol=str(sym),
                    period="daily",
                    start_date=s_start,
                    end_date=s_end,
                    adjust="qfq",
                )
                if df is None or df.empty:
                    continue
                df = df.rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "最高": "high",
                        "最低": "low",
                        "收盘": "close",
                        "成交量": "volume",
                        "成交额": "amount",
                    }
                )
                df["symbol"] = str(sym)
                df["date"] = pd.to_datetime(df["date"])
                df["adj_close"] = df["close"]
                df["market_cap"] = np.nan
                df["shares_outstanding"] = np.nan
                df["industry"] = "Unknown"
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"akshare price fetch failed for {sym}: {exc}")
        if not frames:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in PRICE_COLUMNS})
        out = pd.concat(frames, ignore_index=True)
        return out[[c for c in PRICE_COLUMNS if c in out.columns]]

    def fetch_fundamentals(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        log.warning("AkShare fundamentals adapter: point-in-time release dates are approximated")
        rows = []
        for sym in symbols:
            try:
                df = self._ak.stock_financial_abstract(symbol=str(sym))
                if df is None or df.empty:
                    continue
                for _, r in df.iterrows():
                    period_end = pd.to_datetime(r.get("报告期"), errors="coerce")
                    if pd.isna(period_end):
                        continue

                    def g(key, r=r):
                        return pd.to_numeric(r.get(key), errors="coerce")

                    rows.append(
                        {
                            "symbol": str(sym),
                            "fiscal_period": str(pd.Period(period_end, freq="Q")),
                            "period_end": period_end,
                            "report_date": period_end + pd.Timedelta(days=90),
                            "revenue": g("营业总收入"),
                            "cogs": g("营业成本"),
                            "gross_profit": np.nan,
                            "net_income": g("归母净利润"),
                            "total_assets": g("资产总计"),
                            "total_equity": g("股东权益合计"),
                            "total_debt": np.nan,
                            "operating_cashflow": np.nan,
                            "capex": np.nan,
                            "ebit": np.nan,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning(f"akshare fundamentals failed for {sym}: {exc}")
        if not rows:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in FUNDAMENTAL_COLUMNS})
        df = pd.DataFrame(rows)
        return df[[c for c in FUNDAMENTAL_COLUMNS if c in df.columns] + ["period_end"]]

    def fetch_constituents(
        self, index_id: str = "000300", start: str = "", end: str = ""
    ) -> pd.DataFrame:
        try:
            df = self._ak.index_stock_cons(symbol=index_id)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"akshare constituents failed for {index_id}: {exc}")
            return pd.DataFrame({c: pd.Series(dtype="object") for c in CONSTITUENT_COLUMNS})
        cols = df.columns
        sym_col = next((c for c in cols if "代码" in c), cols[1])
        date_col = next((c for c in cols if "日期" in c), None)
        date = pd.Timestamp(date_col and df[date_col].iloc[0] or pd.Timestamp.today())
        syms = df[sym_col].astype(str).str.zfill(6).tolist()
        return pd.DataFrame(
            {
                "date": date,
                "symbol": syms,
                "index_id": index_id,
                "weight": 1.0 / max(len(syms), 1),
            }
        )[CONSTITUENT_COLUMNS]

    def fetch_macro(self, series: Sequence[str], start: str, end: str) -> pd.DataFrame:
        mapping = {
            "CPI_YOY": ("macro_china_cpi_monthly", "全国-当月"),
            "PPI_YOY": ("macro_china_ppi", "当月同比"),
            "M2_YOY": ("macro_china_money_supply", "M2-同比增长"),
        }
        rows = []
        for sid in series:
            fn_name, *_ = mapping.get(sid, (None,))
            if not fn_name:
                continue
            try:
                fn = getattr(self._ak, fn_name)
                df = fn()
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "date": pd.to_datetime(r.iloc[0]),
                            "series_id": sid,
                            "value": float(r.iloc[1]),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning(f"akshare macro failed for {sid}: {exc}")
        if not rows:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in MACRO_COLUMNS})
        return pd.DataFrame(rows)[MACRO_COLUMNS]

    def fetch_industry(self, symbols: Sequence[str]) -> pd.DataFrame:
        try:
            df = self._ak.stock_board_industry_cons_em()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"akshare industry map failed: {exc}")
            return pd.DataFrame(columns=["symbol", "industry"])
        sym_col = next((c for c in df.columns if "代码" in c), None)
        ind_col = next((c for c in df.columns if "行业" in c or "板块" in c), None)
        if sym_col is None or ind_col is None:
            return pd.DataFrame(columns=["symbol", "industry"])
        out = df[[sym_col, ind_col]].rename(columns={sym_col: "symbol", ind_col: "industry"})
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)
        return out[out["symbol"].isin([str(s) for s in symbols])]


__all__ = [
    "YahooFinanceProvider",
    "AkShareProvider",
    "ProviderUnavailableError",
]

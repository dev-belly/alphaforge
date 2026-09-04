"""Offline tests for the live-data vendor adapters (``vendors.py``).

The adapters need network access, so the *live* path stays untested here — but
their parsing, column-mapping, error-handling and graceful-degradation logic is
pure and is worth locking down. Everything below runs with **no network**:

  * ``yfinance`` / ``requests`` are replaced in ``sys.modules`` with fakes, so
    the tests are deterministic regardless of what the CI image has installed.
  * HTTP is never issued: the fakes record the request and return canned JSON.

Contract under test: every fetch method must return a well-formed frame with the
canonical column schema even when the vendor fails, so the pipeline degrades
instead of crashing.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphaforge.data.providers.base import (
    CONSTITUENT_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    MACRO_COLUMNS,
    PRICE_COLUMNS,
)
from alphaforge.data.providers.vendors import (
    EastMoneyProvider,
    ProviderUnavailableError,
    YahooFinanceProvider,
)

# --------------------------------------------------------------------------
# Fake `requests` (EastMoney): records calls, returns canned JSON.
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeRequests:
    """Minimal ``requests`` stand-in. Payloads are popped in order."""

    def __init__(self, *payloads: dict[str, Any] | Exception) -> None:
        self._payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: int | None = None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        nxt = self._payloads.pop(0) if self._payloads else {"data": None}
        if isinstance(nxt, Exception):
            raise nxt
        return _FakeResponse(nxt)


def _eastmoney(payloads: dict[str, Any] | Exception) -> EastMoneyProvider:
    """Build an EastMoney provider whose HTTP layer is a recorded fake."""
    fake = _FakeRequests(*payloads)
    monkeypatched = types.SimpleNamespace(get=fake.get, Session=lambda *a, **k: None)
    original = sys.modules.get("requests")
    sys.modules["requests"] = monkeypatched  # type: ignore[assignment]
    try:
        provider = EastMoneyProvider()
    finally:
        if original is None:
            sys.modules.pop("requests", None)
        else:
            sys.modules["requests"] = original
    provider._requests = fake
    provider._fake = fake  # type: ignore[attr-defined]
    return provider


# --------------------------------------------------------------------------
# EastMoney: pure helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("600519", "1.600519"),  # Shanghai main board
        ("601318", "1.601318"),
        ("688981", "1.688981"),  # Shanghai STAR market
        ("000001", "0.000001"),  # Shenzhen main board
        ("300750", "0.300750"),  # Shenzhen ChiNext
        ("002594", "0.002594"),
        ("430047", "0.430047"),  # Beijing Stock Exchange
        ("1", "0.000001"),  # un-padded code is zero-filled first
        ("510300", "1.510300"),  # Shanghai ETF -> default Shanghai branch
    ],
)
def test_eastmoney_secid_maps_code_to_market(code: str, expected: str) -> None:
    assert EastMoneyProvider._secid(code) == expected


@pytest.mark.parametrize(
    ("index_id", "expected"),
    [
        ("000300", "1.000300"),  # CSI 300 -> Shanghai
        ("000905", "1.000905"),  # CSI 500 -> Shanghai
        ("399001", "0.399001"),  # Shenzhen Component
        ("399006", "0.399006"),  # ChiNext index
        ("1.000300", "1.000300"),  # already a secid -> passthrough
    ],
)
def test_eastmoney_index_secid_maps_index_to_market(index_id: str, expected: str) -> None:
    assert EastMoneyProvider._index_secid(index_id) == expected


# --------------------------------------------------------------------------
# EastMoney: HTTP layer + price parsing
# --------------------------------------------------------------------------

_KLINE_ROWS = [
    "2024-01-02,10.0,10.5,10.8,9.9,1000",
    "2024-01-03,10.5,11.0,11.2,10.4,1200",
]


def test_eastmoney_klines_builds_request_and_parses_payload() -> None:
    provider = _eastmoney([{"data": {"klines": list(_KLINE_ROWS)}}])
    klines = provider._klines("1.600519", "2024-01-01", "2024-01-31")

    assert klines == _KLINE_ROWS
    call = provider._fake.calls[0]  # type: ignore[attr-defined]
    assert call["url"] == EastMoneyProvider._KLINE_URL
    # EastMoney wants YYYYMMDD (no dashes), daily bars (klt=101), qfq (fqt=1).
    assert call["params"]["secid"] == "1.600519"
    assert call["params"]["beg"] == "20240101"
    assert call["params"]["end"] == "20240131"
    assert call["params"]["klt"] == "101"
    assert call["params"]["fqt"] == "1"


@pytest.mark.parametrize("payload", [{"data": None}, {"data": {"klines": []}}, {}])
def test_eastmoney_klines_empty_payload_returns_empty(payload: dict[str, Any]) -> None:
    provider = _eastmoney([payload])
    assert provider._klines("1.600519", "2024-01-01", "2024-01-31") == []


def test_eastmoney_klines_propagates_http_error() -> None:
    provider = _eastmoney([{"data": {"klines": list(_KLINE_ROWS)}}])
    provider._requests = _FakeRequests(RuntimeError("503 boom"))
    with pytest.raises(RuntimeError, match="503 boom"):
        provider._klines("1.600519", "2024-01-01", "2024-01-31")


def test_eastmoney_fetch_prices_parses_ohlcv(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _eastmoney([{"data": {"klines": list(_KLINE_ROWS)}}])
    monkeypatch.setattr(provider, "_klines", lambda *a, **k: list(_KLINE_ROWS))

    out = provider.fetch_prices(["600519"], "2024-01-01", "2024-01-31")

    assert list(out.columns) == PRICE_COLUMNS
    assert len(out) == 2
    assert out["symbol"].tolist() == ["600519", "600519"]
    assert out["close"].tolist() == [10.5, 11.0]  # p[2] is close, not open
    assert out["open"].tolist() == [10.0, 10.5]
    assert out["high"].tolist() == [10.8, 11.2]
    assert out["low"].tolist() == [9.9, 10.4]
    assert out["volume"].tolist() == [1000.0, 1200.0]
    # qfq-adjusted close mirrors close; fundamentals are unavailable by design.
    pd.testing.assert_series_equal(out["adj_close"], out["close"], check_names=False)
    assert out["market_cap"].isna().all()
    assert (out["industry"] == "Unknown").all()
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_eastmoney_fetch_prices_all_malformed_rows_yields_empty_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _eastmoney([{"data": None}])
    monkeypatch.setattr(provider, "_klines", lambda *a, **k: ["junk", "also,junk"])
    out = provider.fetch_prices(["600519"], "2024-01-01", "2024-01-31")
    assert list(out.columns) == PRICE_COLUMNS and out.empty


def test_eastmoney_fetch_prices_skips_malformed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _eastmoney([{"data": None}])
    monkeypatch.setattr(
        provider,
        "_klines",
        lambda *a, **k: ["2024-01-02,10.0,10.5,10.8,9.9,1000", "2024-01-03,broken"],
    )
    out = provider.fetch_prices(["600519"], "2024-01-01", "2024-01-31")
    assert len(out) == 1


def test_eastmoney_fetch_prices_degrades_on_empty_or_failed_vendor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _eastmoney([{"data": None}])

    monkeypatch.setattr(provider, "_klines", lambda *a, **k: [])
    empty = provider.fetch_prices(["600519"], "2024-01-01", "2024-01-31")
    assert list(empty.columns) == PRICE_COLUMNS and empty.empty

    def _boom(*a: object, **k: object) -> list[str]:
        raise RuntimeError("connection reset")

    monkeypatch.setattr(provider, "_klines", _boom)
    failed = provider.fetch_prices(["600519"], "2024-01-01", "2024-01-31")
    assert list(failed.columns) == PRICE_COLUMNS and failed.empty


def test_eastmoney_benchmark_prices_uses_index_secid_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _eastmoney([{"data": {"klines": list(_KLINE_ROWS)}}])
    seen: list[str] = []

    def _fake_klines(secid: str, start: str, end: str) -> list[str]:
        seen.append(secid)
        return list(_KLINE_ROWS)

    monkeypatch.setattr(provider, "_klines", _fake_klines)
    bench = provider.benchmark_prices("000300", "2024-01-01", "2024-01-31")

    assert seen == ["1.000300"]
    assert bench.name == "benchmark"
    assert isinstance(bench.index, pd.DatetimeIndex)
    assert bench.tolist() == [10.5, 11.0]


def test_eastmoney_benchmark_prices_skips_malformed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _eastmoney([{"data": None}])
    monkeypatch.setattr(
        provider,
        "_klines",
        lambda *a, **k: ["2024-01-02,10.0,10.5,10.8,9.9,1000", "2024-01-03,broken"],
    )
    bench = provider.benchmark_prices("000300", "2024-01-01", "2024-01-31")
    assert bench.tolist() == [10.5]


def test_eastmoney_benchmark_prices_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _eastmoney([{"data": None}])
    monkeypatch.setattr(provider, "_klines", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="EastMoney benchmark unavailable for 000300"):
        provider.benchmark_prices("000300", "2024-01-01", "2024-01-31")


def test_eastmoney_unsupported_endpoints_return_empty_canonical_frames() -> None:
    provider = _eastmoney([{"data": None}])
    for frame, columns in (
        (provider.fetch_fundamentals(["600519"], "2024-01-01", "2024-01-31"), FUNDAMENTAL_COLUMNS),
        (provider.fetch_constituents("000300", "2024-01-01", "2024-01-31"), CONSTITUENT_COLUMNS),
        (provider.fetch_macro(["VIX"], "2024-01-01", "2024-01-31"), MACRO_COLUMNS),
    ):
        assert list(frame.columns) == columns and frame.empty

    ind = provider.fetch_industry(["600519", "000001"])
    assert list(ind.columns) == ["symbol", "industry"]
    assert ind["industry"].tolist() == ["Unknown", "Unknown"]


# --------------------------------------------------------------------------
# Fake `yfinance` (Yahoo): dispatch on list vs. scalar ticker.
# --------------------------------------------------------------------------

_DATES = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
_OHLCV = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _multiindex_frame(tickers: list[str]) -> pd.DataFrame:
    n = len(tickers) * len(_OHLCV)
    data = np.arange(len(_DATES) * n, dtype=float).reshape(len(_DATES), n)
    cols = pd.MultiIndex.from_product([tickers, _OHLCV])
    return pd.DataFrame(data, index=_DATES, columns=cols)


def _flat_frame() -> pd.DataFrame:
    data = np.arange(len(_DATES) * len(_OHLCV), dtype=float).reshape(len(_DATES), len(_OHLCV))
    return pd.DataFrame(data, index=_DATES, columns=_OHLCV)


class _FastInfo:
    def __init__(self, shares: float | None) -> None:
        self.shares_outstanding = shares


class _FakeTicker:
    """Fake ``yf.Ticker``. Set ``explode=True`` to simulate a vendor failure."""

    def __init__(
        self,
        symbol: str,
        shares: float | None = 1_000.0,
        explode: bool = False,
        info: dict[str, Any] | None = None,
        statements: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.symbol = symbol
        self._shares = shares
        self._explode = explode
        self._info = info
        self._statements = statements or {}

    @property
    def fast_info(self) -> _FastInfo:
        if self._explode:
            raise RuntimeError(f"fast_info unavailable for {self.symbol}")
        return _FastInfo(self._shares)

    @property
    def info(self) -> dict[str, Any]:
        if self._explode:
            raise RuntimeError("info unavailable")
        return self._info if self._info is not None else {}

    @property
    def quarterly_balance_sheet(self) -> pd.DataFrame:
        return self._statements.get("bs", pd.DataFrame())

    @property
    def quarterly_financials(self) -> pd.DataFrame:
        return self._statements.get("inc", pd.DataFrame())

    @property
    def quarterly_cashflow(self) -> pd.DataFrame:
        return self._statements.get("cf", pd.DataFrame())


class _ExplodingTicker(_FakeTicker):
    """A ``Ticker`` whose every statement/fundamental access raises."""

    @property
    def quarterly_balance_sheet(self) -> pd.DataFrame:
        raise RuntimeError("statement fetch failed")

    @property
    def quarterly_financials(self) -> pd.DataFrame:
        raise RuntimeError("statement fetch failed")

    @property
    def quarterly_cashflow(self) -> pd.DataFrame:
        raise RuntimeError("statement fetch failed")


class _FakeYFinance:
    def __init__(
        self,
        price_frame: pd.DataFrame | None = None,
        ticker_factory: Any = None,
        scalar_frame: pd.DataFrame | None = None,
        raise_on_download: str | None = None,
    ) -> None:
        self._price_frame = price_frame
        self._scalar_frame = scalar_frame if scalar_frame is not None else _flat_frame()
        self._ticker_factory = ticker_factory or (lambda sym: _FakeTicker(sym))
        self._raise_on_download = raise_on_download
        self.download_calls: list[dict[str, Any]] = []

    def download(self, tickers: Any, **kwargs: Any) -> pd.DataFrame:
        self.download_calls.append({"tickers": tickers, **kwargs})
        if self._raise_on_download is not None and tickers == self._raise_on_download:
            raise RuntimeError(f"download failed for {tickers}")
        # Scalar ticker == macro / benchmark request; list == price panel.
        if isinstance(tickers, str):
            return self._scalar_frame
        return (
            self._price_frame if self._price_frame is not None else _multiindex_frame(list(tickers))
        )

    def Ticker(self, symbol: str) -> Any:  # noqa: N802 - mirrors yfinance's API
        return self._ticker_factory(symbol)


def _yahoo(monkeypatch: pytest.MonkeyPatch, fake: _FakeYFinance) -> YahooFinanceProvider:
    """Build a Yahoo provider bound to a fake ``yfinance`` module.

    Injecting ``sys.modules['yfinance']`` makes this deterministic whether or
    not the optional dependency happens to be installed.
    """
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    provider = YahooFinanceProvider()
    assert provider._yf is fake
    return provider


def test_yahoo_raises_actionable_error_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``sys.modules[name] = None`` makes ``import name`` raise ImportError.
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ProviderUnavailableError, match="pip install yfinance"):
        YahooFinanceProvider()


def test_provider_unavailable_error_is_runtime_error() -> None:
    assert issubclass(ProviderUnavailableError, RuntimeError)


# --------------------------------------------------------------------------
# Yahoo: price / fundamentals / macro / industry / benchmark
# --------------------------------------------------------------------------


def test_yahoo_fetch_prices_parses_multiindex_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeYFinance(price_frame=_multiindex_frame(["AAPL", "MSFT"]))
    provider = _yahoo(monkeypatch, fake)

    out = provider.fetch_prices(["AAPL", "MSFT"], "2024-01-01", "2024-01-31")

    assert list(out.columns) == PRICE_COLUMNS
    assert set(out["symbol"]) == {"AAPL", "MSFT"}
    assert len(out) == 4
    # Renaming must be applied: raw Yahoo headers are Title case.
    assert not {"Open", "Close", "Adj Close"} & set(out.columns)
    # market_cap = shares_outstanding * last close (1000 * close).
    aapl = out[out["symbol"] == "AAPL"]
    assert aapl["market_cap"].iloc[0] == pytest.approx(1000.0 * aapl["close"].iloc[-1])
    assert out["shares_outstanding"].isna().all()
    assert (out["industry"] == "Unknown").all()


def test_yahoo_fetch_prices_single_ticker_flat_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeYFinance(price_frame=_flat_frame())
    provider = _yahoo(monkeypatch, fake)

    out = provider.fetch_prices(["AAPL"], "2024-01-01", "2024-01-31")

    assert list(out.columns) == PRICE_COLUMNS
    assert len(out) == 2
    assert (out["symbol"] == "AAPL").all()


def test_yahoo_fetch_prices_skips_symbols_absent_from_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delisted/unknown ticker must not fabricate rows."""
    fake = _FakeYFinance(price_frame=_multiindex_frame(["AAPL"]))
    provider = _yahoo(monkeypatch, fake)

    out = provider.fetch_prices(["NOT_A_TICKER"], "2024-01-01", "2024-01-31")

    assert list(out.columns) == PRICE_COLUMNS
    assert out.empty


def test_yahoo_fetch_prices_rejects_flat_panel_for_multi_ticker_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-ticker frame must never be broadcast across every symbol.

    If Yahoo collapses to flat columns we cannot tell which symbol the rows
    belong to, so the adapter must drop them rather than mislabel the panel.
    """
    provider = _yahoo(monkeypatch, _FakeYFinance(price_frame=_flat_frame()))
    out = provider.fetch_prices(["AAPL", "MSFT"], "2024-01-01", "2024-01-31")
    assert list(out.columns) == PRICE_COLUMNS and out.empty


def test_yahoo_market_cap_is_nan_when_fast_info_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeYFinance(
        price_frame=_multiindex_frame(["AAPL"]),
        ticker_factory=lambda sym: _FakeTicker(sym, explode=True),
    )
    provider = _yahoo(monkeypatch, fake)

    out = provider.fetch_prices(["AAPL"], "2024-01-01", "2024-01-31")

    assert out["market_cap"].isna().all()
    assert len(out) == 2  # the price panel is still returned


def test_yahoo_fundamentals_applies_conservative_publication_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    period = pd.Timestamp("2024-03-31")
    bs = pd.DataFrame(
        {
            period: {
                "Total Assets": 100.0,
                "Stockholders Equity": 60.0,
                "Total Debt": 20.0,
            }
        }
    )
    # Deliberately uses "Revenue" (not "Total Revenue") to exercise the alias
    # fallback, and omits cogs to prove missing items become NaN.
    inc = pd.DataFrame(
        {period: {"Revenue": 30.0, "Gross Profit": 12.0, "EBIT": 8.0, "Net Income": 5.0}}
    )
    cf = pd.DataFrame({period: {"Operating Cash Flow": 9.0, "Capital Expenditures": -2.0}})
    fake = _FakeYFinance(
        ticker_factory=lambda sym: _FakeTicker(sym, statements={"bs": bs, "inc": inc, "cf": cf})
    )
    provider = _yahoo(monkeypatch, fake)

    out = provider.fetch_fundamentals(["AAPL"], "2024-01-01", "2024-12-31")

    assert list(out.columns) == FUNDAMENTAL_COLUMNS + ["period_end"]
    row = out.iloc[0]
    assert row["fiscal_period"] == "2024Q1"
    assert row["period_end"] == period
    # Yahoo exposes no release dates -> adapter adds the configured lag.
    assert row["report_date"] == period + pd.Timedelta(days=90)
    assert row["revenue"] == pytest.approx(30.0)
    assert row["gross_profit"] == pytest.approx(12.0)
    assert row["ebit"] == pytest.approx(8.0)
    assert row["net_income"] == pytest.approx(5.0)
    assert row["total_assets"] == pytest.approx(100.0)
    assert row["total_equity"] == pytest.approx(60.0)
    assert row["total_debt"] == pytest.approx(20.0)
    assert row["operating_cashflow"] == pytest.approx(9.0)
    assert row["capex"] == pytest.approx(-2.0)
    assert pd.isna(row["cogs"])

    # The lag is configurable.
    lag_provider = YahooFinanceProvider(fundamental_lag_days=45)
    assert lag_provider.fundamental_lag_days == 45


def test_yahoo_fundamentals_missing_statements_become_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partially-reporting vendor yields NaNs for absent statements, not a crash."""
    period = pd.Timestamp("2024-03-31")
    bs = pd.DataFrame({period: {"Total Assets": 100.0}})  # balance sheet only
    provider = _yahoo(
        monkeypatch,
        _FakeYFinance(ticker_factory=lambda sym: _FakeTicker(sym, statements={"bs": bs})),
    )

    out = provider.fetch_fundamentals(["AAPL"], "2024-01-01", "2024-12-31")

    assert len(out) == 1
    row = out.iloc[0]
    assert row["total_assets"] == pytest.approx(100.0)  # present
    for field in ("revenue", "net_income", "operating_cashflow", "capex"):
        assert pd.isna(row[field])  # absent income/cashflow statements


def test_yahoo_fundamentals_degrade_when_vendor_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # Case 1: the statement access itself raises.
    provider = _yahoo(monkeypatch, _FakeYFinance(ticker_factory=_ExplodingTicker))
    out = provider.fetch_fundamentals(["AAPL"], "2024-01-01", "2024-12-31")
    assert list(out.columns) == FUNDAMENTAL_COLUMNS and out.empty

    # Case 2: the vendor returns an empty balance sheet -> nothing to report.
    provider2 = _yahoo(monkeypatch, _FakeYFinance(ticker_factory=lambda sym: _FakeTicker(sym)))
    out2 = provider2.fetch_fundamentals(["AAPL"], "2024-01-01", "2024-12-31")
    assert list(out2.columns) == FUNDAMENTAL_COLUMNS and out2.empty


def test_yahoo_macro_maps_series_ids_and_skips_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    fake = _FakeYFinance(scalar_frame=pd.DataFrame({"Adj Close": [17.5, 18.0]}, index=idx))
    provider = _yahoo(monkeypatch, fake)

    out = provider.fetch_macro(["VIX", "NOT_A_SERIES"], "2024-01-01", "2024-01-31")

    assert list(out.columns) == MACRO_COLUMNS
    assert out["series_id"].tolist() == ["VIX", "VIX"]
    assert out["value"].tolist() == [17.5, 18.0]
    # The VIX id must be translated to Yahoo's ticker, not passed through raw.
    assert fake.download_calls[0]["tickers"] == "^VIX"

    empty = provider.fetch_macro(["NOPE"], "2024-01-01", "2024-01-31")
    assert list(empty.columns) == MACRO_COLUMNS and empty.empty

    # A mapped-but-failing download must degrade, not crash.
    flaky = _yahoo(monkeypatch, _FakeYFinance(raise_on_download="^VIX"))
    assert list(flaky.fetch_macro(["VIX"], "2024-01-01", "2024-01-31").columns) == MACRO_COLUMNS


def test_yahoo_industry_prefers_sector_then_industry(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(sym: str) -> _FakeTicker:
        if sym == "AAPL":
            return _FakeTicker(
                sym, info={"sector": "Technology", "industry": "Consumer Electronics"}
            )
        if sym == "MSFT":
            return _FakeTicker(sym, info={"industry": "Software"})  # no sector key
        return _FakeTicker(sym, explode=True)  # vendor failure

    provider = _yahoo(monkeypatch, _FakeYFinance(ticker_factory=factory))
    out = provider.fetch_industry(["AAPL", "MSFT", "BROKEN"])

    assert list(out.columns) == ["symbol", "industry"]
    assert out.set_index("symbol")["industry"].to_dict() == {
        "AAPL": "Technology",
        "MSFT": "Software",
        "BROKEN": "Unknown",
    }


def test_yahoo_constituents_returns_current_snapshot_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Yahoo has no historical membership -> weights are equal, not point-in-time."""
    table = pd.DataFrame({"Symbol": ["AAPL", "BRK.B", "BF.B"]})
    monkeypatch.setattr(pd, "read_html", lambda *a, **k: [table])
    provider = _yahoo(monkeypatch, _FakeYFinance())

    out = provider.fetch_constituents("SP500", "2024-01-01", "2024-06-30")

    assert list(out.columns) == CONSTITUENT_COLUMNS
    assert set(out["symbol"]) == {"AAPL", "BRK-B", "BF-B"}  # dots -> dashes
    assert out["weight"].nunique() == 1
    assert out["weight"].iloc[0] == pytest.approx(1.0 / 3.0)
    assert (out["index_id"] == "SP500").all()


def test_yahoo_constituents_degrade_when_scrape_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a: object, **k: object) -> list[pd.DataFrame]:
        raise RuntimeError("no network")

    monkeypatch.setattr(pd, "read_html", _boom)
    provider = _yahoo(monkeypatch, _FakeYFinance())

    out = provider.fetch_constituents("SP500", "2024-01-01", "2024-06-30")

    assert list(out.columns) == CONSTITUENT_COLUMNS and out.empty


def test_yahoo_benchmark_prices_returns_named_adj_close_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    fake = _FakeYFinance(scalar_frame=pd.DataFrame({"Adj Close": [100.0, 101.5]}, index=idx))
    provider = _yahoo(monkeypatch, fake)

    bench = provider.benchmark_prices("^GSPC", "2024-01-01", "2024-01-31")

    assert bench.name == "benchmark"
    assert bench.tolist() == [100.0, 101.5]
    assert fake.download_calls[0]["tickers"] == "^GSPC"

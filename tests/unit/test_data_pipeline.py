"""Offline tests for the ETL orchestration layer (``data/pipeline.py``).

``DataPipeline.run()`` is the single entry point behind the CLI, API, notebooks
and dashboards, so a defect here is silently inherited by every consumer. These
tests drive it with a **controllable fake provider** (no network, no real
vendor) and lock the contracts that matter:

  * Universe resolution: an explicit symbol list always wins; each provider
    falls back to its own curated universe; an unknown provider fails loudly.
  * An empty price panel is a hard error, never an empty downstream dataset.
  * ``benchmark`` leaves the pipeline as a daily **return** series (not price
    levels) so the backtester, attribution and regime model cannot
    double-difference it — the bug fixed in ``test_benchmark_no_double_diff``.
  * Persisting is optional and writes the quality report the README advertises.

Everything writes into ``tmp_path``; the default-``DataStore`` branch is
exercised under ``monkeypatch.chdir`` so it can never touch the repo.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.data.pipeline import (
    DataPipeline,
    _fetch_benchmark_returns,
    _resolve_symbols,
    _universe_long,
    load_bundle,
)
from alphaforge.data.providers.base import (
    CONSTITUENT_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    MACRO_COLUMNS,
    DataProvider,
)
from alphaforge.data.providers.local import LocalParquetProvider
from alphaforge.data.providers.vendors import AKSHARE_DEFAULT_UNIVERSE, YAHOO_DEFAULT_UNIVERSE
from alphaforge.data.storage import DataStore

SYMBOLS = ("AAA", "BBB")


def _panel(
    days: int = 70,
    symbols: Sequence[str] = SYMBOLS,
    industry: str | None = "Information Technology",
) -> pd.DataFrame:
    """A clean, eligible long-format panel (positive prices, liquid, enough history)."""
    dates = pd.bdate_range("2024-01-02", periods=days)
    frames = []
    for i, sym in enumerate(symbols):
        close = 100.0 + i * 10.0 + np.arange(days, dtype=float) * 0.05
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "adj_close": close,
                    "volume": 2e5,  # -> ADV well above the liquidity screen
                    "market_cap": close * 1e6,
                    "shares_outstanding": 1e6,
                    "industry": industry,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class _FakeProvider(DataProvider):
    """Deterministic provider: whatever the test injects is what the ETL sees."""

    def __init__(
        self,
        name: str = "fake",
        prices: pd.DataFrame | None = None,
        industry: pd.DataFrame | None = None,
        benchmark: pd.Series | None = None,
        no_benchmark: bool = False,
    ) -> None:
        self.name = name
        self._prices = _panel() if prices is None else prices
        self._industry = (
            industry
            if industry is not None
            else pd.DataFrame({"symbol": list(SYMBOLS), "industry": "Information Technology"})
        )
        self._benchmark = benchmark
        self._no_benchmark = no_benchmark
        self.calls: list[dict] = []

    def fetch_prices(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        self.calls.append({"method": "prices", "symbols": list(symbols)})
        return self._prices.copy()

    def fetch_fundamentals(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        return self.empty_frame(FUNDAMENTAL_COLUMNS)

    def fetch_constituents(self, index_id: str, start: str, end: str) -> pd.DataFrame:
        return self.empty_frame(CONSTITUENT_COLUMNS)

    def fetch_macro(self, series: Sequence[str], start: str, end: str) -> pd.DataFrame:
        return self.empty_frame(MACRO_COLUMNS)

    def fetch_industry(self, symbols: Sequence[str]) -> pd.DataFrame:
        return self._industry.copy()

    def benchmark_prices(self, index_id: str, start: str, end: str) -> pd.Series:
        if self._no_benchmark:
            raise NotImplementedError(f"{self.name} does not provide benchmark series")
        if self._benchmark is None:
            raise NotImplementedError(f"{self.name} does not provide benchmark series")
        return self._benchmark.copy()

    def symbols(self) -> list[str]:
        return list(SYMBOLS)


class _OpaqueProvider(_FakeProvider):
    """A provider that cannot declare a universe (mirrors the base contract)."""

    def symbols(self) -> list[str]:  # type: ignore[override]
        raise NotImplementedError(f"{self.name} does not expose a fixed symbol list")


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    return DataStore(root=tmp_path / "processed", use_duckdb=False)


# --------------------------------------------------------------------------
# Universe resolution
# --------------------------------------------------------------------------


def test_explicit_symbols_always_win() -> None:
    provider = _FakeProvider(name="yahoo")
    assert _resolve_symbols(provider, ["TSLA"], "SP500") == ["TSLA"]


def test_sample_provider_resolves_its_own_cross_section() -> None:
    provider = _FakeProvider(name="sample")
    assert _resolve_symbols(provider, None, "SP500_SAMPLE") == list(SYMBOLS)


def test_yahoo_falls_back_to_curated_universe() -> None:
    assert _resolve_symbols(_FakeProvider(name="yahoo"), None, "SP500") == YAHOO_DEFAULT_UNIVERSE


@pytest.mark.parametrize("name", ["akshare", "eastmoney"])
def test_ashare_providers_fall_back_to_curated_universe(name: str) -> None:
    resolved = _resolve_symbols(_FakeProvider(name=name), None, "000300")
    assert resolved == AKSHARE_DEFAULT_UNIVERSE


def test_unsupported_provider_raises() -> None:
    """A backend that can neither be mapped nor declare a universe fails loudly."""
    with pytest.raises(ValueError, match="Unsupported data provider: 'tushare'"):
        _resolve_symbols(_OpaqueProvider(name="tushare"), None, "000300")


def test_local_backend_declares_its_persisted_universe(tmp_path: Path) -> None:
    """Regression: the default production backend used to be rejected by the ETL.

    ``_resolve_symbols`` only knew the sample and live vendors, so
    ``DataPipeline(provider="local")`` raised ``Unsupported data provider:
    'local'`` and the documented production path could not run unconfigured.
    """
    root = tmp_path / "processed"
    store = DataStore(root=root, use_duckdb=False)
    store.upsert_prices(_panel())

    provider = LocalParquetProvider(root=root)
    assert provider.symbols() == sorted(SYMBOLS)

    result = DataPipeline(provider=provider, store=store).run(persist=False)
    assert set(result.bundle.prices["symbol"]) == set(SYMBOLS)
    assert result.bundle.metadata["provider"] == "local"


def test_local_backend_without_artifacts_has_an_empty_universe(tmp_path: Path) -> None:
    assert LocalParquetProvider(root=tmp_path).symbols() == []


def test_empty_symbol_list_is_treated_as_unset() -> None:
    """``[]`` must fall through to the provider universe, not resolve to no symbols."""
    assert _resolve_symbols(_FakeProvider(name="sample"), [], "SP500_SAMPLE") == list(SYMBOLS)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_pipeline_accepts_a_provider_instance_or_a_name(store: DataStore) -> None:
    fake = _FakeProvider()
    assert DataPipeline(provider=fake, store=store).provider is fake
    assert DataPipeline(provider="sample", store=store).provider.name == "sample"


def test_pipeline_defaults_its_store_without_touching_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default ``DataStore()`` points at ./data/processed - pin it to a temp CWD."""
    monkeypatch.chdir(tmp_path)
    pipeline = DataPipeline(provider="sample")
    # The default root is relative, so it follows the CWD rather than the repo.
    assert pipeline.store.root == Path("data") / "processed"
    assert (tmp_path / "data" / "processed").is_dir()


# --------------------------------------------------------------------------
# run()
# --------------------------------------------------------------------------


def test_run_raises_when_the_panel_is_empty(store: DataStore) -> None:
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in ["date", "symbol", "close"]})
    pipeline = DataPipeline(provider=_FakeProvider(prices=empty), store=store)

    with pytest.raises(RuntimeError, match="returned an empty price panel"):
        pipeline.run(start="2024-01-01", end="2024-06-30")


def test_run_returns_a_bundle_quality_and_universe(store: DataStore) -> None:
    result = DataPipeline(provider=_FakeProvider(), store=store).run(persist=False)

    assert not result.bundle.prices.empty
    assert set(result.bundle.prices["symbol"]) == set(SYMBOLS)
    assert result.bundle.metadata["provider"] == "fake"
    assert result.bundle.metadata["provenance"] == "FAKE"
    assert "survivorship" in result.bundle.metadata
    # Universe is a (dates x symbols) boolean panel.
    assert list(result.universe.columns) == list(SYMBOLS)
    assert result.universe.dtypes.eq(bool).all()


def test_run_persists_the_advertised_tables(store: DataStore) -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    provider = _FakeProvider(benchmark=pd.Series([100.0, 101.0], index=idx))
    DataPipeline(provider=provider, store=store).run()

    tables = set(store.tables())
    assert {"prices", "universe", "data_quality", "benchmark", "industry"} <= tables
    assert not store.read("prices").empty
    assert not store.read("data_quality").empty


def test_run_skips_persistence_when_disabled(store: DataStore) -> None:
    DataPipeline(provider=_FakeProvider(), store=store).run(persist=False)
    assert store.tables() == []


def test_run_fills_missing_industry_from_the_provider_map(store: DataStore) -> None:
    prices = _panel(industry=None)  # provider supplies no classification
    industry = pd.DataFrame({"symbol": list(SYMBOLS), "industry": ["Energy", "Utilities"]})
    result = DataPipeline(
        provider=_FakeProvider(prices=prices, industry=industry), store=store
    ).run(persist=False)

    mapping = result.bundle.prices.groupby("symbol")["industry"].first()
    assert mapping.to_dict() == {"AAA": "Energy", "BBB": "Utilities"}


def test_summary_reports_rows_symbols_and_span(store: DataStore) -> None:
    result = DataPipeline(provider=_FakeProvider(), store=store).run(persist=False)
    summary = result.summary()

    assert summary["rows"] == len(result.bundle.prices)
    assert summary["symbols"] == len(SYMBOLS)
    assert summary["start"] == "2024-01-02"
    assert isinstance(summary["quality"], dict)


# --------------------------------------------------------------------------
# Benchmark handling - returns, never levels
# --------------------------------------------------------------------------


def test_run_exposes_benchmark_as_daily_returns(store: DataStore) -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    provider = _FakeProvider(benchmark=pd.Series([100.0, 110.0, 99.0], index=idx))
    result = DataPipeline(provider=provider, store=store).run(persist=False)

    bench = result.bundle.benchmark
    assert bench is not None
    assert bench.name == "benchmark"
    # Levels 100 -> 110 -> 99 become returns 0.10 and -0.10 (first bar dropped).
    assert bench.tolist() == pytest.approx([0.10, -0.10])
    assert bench.index.is_monotonic_increasing


def test_run_tolerates_a_provider_without_a_benchmark(store: DataStore) -> None:
    result = DataPipeline(provider=_FakeProvider(no_benchmark=True), store=store).run(persist=False)
    assert result.bundle.benchmark is None


@pytest.mark.parametrize(
    ("bench", "expected"),
    [
        (None, None),
        (pd.Series(dtype=float), None),
    ],
)
def test_fetch_benchmark_returns_degrades_to_none(bench: pd.Series | None, expected) -> None:
    provider = _FakeProvider(no_benchmark=True)
    assert _fetch_benchmark_returns(provider, "SP500", "2024-01-01", "2024-06-30") == expected


def test_fetch_benchmark_returns_sorts_and_differences_once() -> None:
    # Deliberately unsorted: the helper must sort before differencing.
    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-04"])
    provider = _FakeProvider(benchmark=pd.Series([110.0, 100.0, 99.0], index=idx))

    rets = _fetch_benchmark_returns(provider, "SP500", "2024-01-01", "2024-06-30")

    assert rets is not None
    assert rets.tolist() == pytest.approx([0.10, -0.10])
    assert rets.name == "benchmark"


# --------------------------------------------------------------------------
# _universe_long / load_bundle
# --------------------------------------------------------------------------


def test_universe_long_keeps_only_members() -> None:
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    universe = pd.DataFrame({"AAA": [True, False], "BBB": [True, True]}, index=dates)

    out = _universe_long(universe)

    assert list(out.columns) == ["date", "symbol"]
    assert len(out) == 3  # (AAA, d0), (BBB, d0), (BBB, d1)
    assert out["symbol"].tolist() == ["AAA", "BBB", "BBB"]


def test_load_bundle_round_trips_a_persisted_run(store: DataStore) -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    provider = _FakeProvider(benchmark=pd.Series([100.0, 110.0, 99.0], index=idx))
    DataPipeline(provider=provider, store=store).run()

    bundle = load_bundle(store.root)

    assert not bundle.prices.empty
    assert set(bundle.prices["symbol"]) == set(SYMBOLS)
    assert bundle.metadata["provenance"] == "PERSISTED"
    assert bundle.benchmark is not None
    assert bundle.benchmark.name == "benchmark"
    assert bundle.benchmark.tolist() == pytest.approx([0.10, -0.10])
    assert bundle.benchmark.index.is_monotonic_increasing


def test_load_bundle_degrades_when_optional_tables_are_absent(tmp_path: Path) -> None:
    """A prices-only store must still load - optional tables degrade to empty frames."""
    root = tmp_path / "processed"
    store = DataStore(root=root, use_duckdb=False)
    store.write("prices", _panel())

    bundle = load_bundle(root)

    assert not bundle.prices.empty
    assert bundle.fundamentals.empty
    assert bundle.macro.empty
    assert bundle.constituents.empty
    assert bundle.benchmark is None  # no persisted benchmark -> no silent zeros

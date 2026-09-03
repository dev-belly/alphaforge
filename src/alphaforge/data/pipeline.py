"""ETL orchestration: provider -> validation -> clean -> store.

``DataPipeline.run()`` is the single entry point used by the CLI, the API, the
notebooks and the tests, so every consumer sees an identically-shaped,
quality-checked dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alphaforge.data.providers import get_provider
from alphaforge.data.providers.base import DataBundle, DataProvider
from alphaforge.data.quality import build_quality_report, clean_prices
from alphaforge.data.storage import DataStore
from alphaforge.data.universe import Universe, UniverseConfig
from alphaforge.utils.logging import Timer, get_logger

log = get_logger("data.pipeline")


@dataclass
class PipelineResult:
    bundle: DataBundle
    quality: object  # DataQualityReport
    universe: pd.DataFrame
    store: DataStore

    def summary(self) -> dict:
        return {
            "rows": len(self.bundle.prices),
            "symbols": int(self.bundle.prices["symbol"].nunique()),
            "start": str(self.bundle.prices["date"].min().date()),
            "end": str(self.bundle.prices["date"].max().date()),
            "quality": self.quality.to_dict(),  # type: ignore[union-attr]
        }


class DataPipeline:
    """Wires a provider, the cleaning rules, quality gates and the store."""

    def __init__(
        self,
        provider: DataProvider | str = "sample",
        store: DataStore | None = None,
        universe_config: UniverseConfig | None = None,
        clip_extreme_returns: float | None = 0.75,
        **provider_kwargs,
    ) -> None:
        self.provider = (
            get_provider(provider, **provider_kwargs) if isinstance(provider, str) else provider
        )
        self.store = store or DataStore()
        self.universe_config = universe_config or UniverseConfig()
        self.clip_extreme_returns = clip_extreme_returns
        log.info(f"DataPipeline provider={self.provider.name}")

    # ------------------------------------------------------------------
    def run(
        self,
        start: str = "2015-01-01",
        end: str = "2024-12-31",
        index_id: str = "SP500_SAMPLE",
        persist: bool = True,
    ) -> PipelineResult:
        with Timer("etl.fetch", log):
            prices = self.provider.fetch_prices(None, start, end)
            fundamentals = self.provider.fetch_fundamentals(None, start, end)
            macro = self.provider.fetch_macro(None, start, end)
            constituents = self.provider.fetch_constituents(index_id, start, end)
            industry = self.provider.fetch_industry(None)

        if prices.empty:
            raise RuntimeError(f"Provider {self.provider.name} returned an empty price panel")

        provenance = str(getattr(prices, "attrs", {}).get("provenance", self.provider.name.upper()))
        with Timer("etl.clean", log):
            prices = clean_prices(prices, clip_extreme_returns=self.clip_extreme_returns)

        with Timer("etl.quality", log):
            quality = build_quality_report(prices, provenance=provenance)
            if not quality.is_acceptable():
                log.warning("Data quality gates tripped - see the persisted report")

        # Attach industry classification where the provider did not supply it.
        if "industry" in prices.columns:
            missing = prices["industry"].isna() if prices["industry"].notna().any() else None
            if missing is not None and missing.any() and not industry.empty:
                mapping = industry.set_index("symbol")["industry"]
                prices.loc[missing, "industry"] = prices.loc[missing, "symbol"].map(mapping)

        # --- benchmark return series ------------------------------------
        # Stored as *daily returns* (not price levels) so the backtest engine,
        # factor attribution and market-regime classification can consume it
        # directly. Providers that cannot supply a benchmark yield None.
        benchmark = _fetch_benchmark_returns(self.provider, index_id, start, end)

        with Timer("etl.universe", log):
            universe = Universe(self.universe_config).build(prices, constituents)

        bundle = DataBundle(
            prices=prices,
            fundamentals=fundamentals,
            macro=macro,
            constituents=constituents,
            metadata={
                "provider": self.provider.name,
                "provenance": provenance,
                "provider_info": self.provider.describe(),
                "survivorship": Universe.survivorship_diagnostics(prices, universe),
                "index_id": index_id,
            },
            benchmark=benchmark,
        )

        if persist:
            with Timer("etl.persist", log):
                self.store.upsert_prices(prices)
                if not fundamentals.empty:
                    self.store.write("fundamentals", fundamentals)
                if not macro.empty:
                    self.store.write("macro", macro)
                if not constituents.empty:
                    self.store.write("constituents", constituents)
                if not industry.empty:
                    self.store.write("industry", industry)
                self.store.write("universe", _universe_long(universe))
                self.store.write("data_quality", quality.to_frame())
                if benchmark is not None and len(benchmark):
                    self.store.write(
                        "benchmark",
                        pd.DataFrame(
                            {"date": benchmark.index, "value": benchmark.to_numpy()}
                        ),
                    )
                self.store.register_views()

        log.info(
            f"ETL complete: {len(prices):,} rows | {prices['symbol'].nunique()} symbols | "
            f"{prices['date'].min().date()} -> {prices['date'].max().date()}"
        )
        return PipelineResult(bundle=bundle, quality=quality, universe=universe, store=self.store)


def _universe_long(universe: pd.DataFrame) -> pd.DataFrame:
    stacked = universe.stack()
    stacked.name = "is_member"
    out = stacked.reset_index()
    out.columns = ["date", "symbol", "is_member"]
    return out[out["is_member"]].drop(columns="is_member")


def load_bundle(root: str | Path = "data/processed") -> DataBundle:
    """Reload a persisted dataset without re-running any vendor call."""
    store = DataStore(root)
    prices = store.read("prices")
    prices["date"] = pd.to_datetime(prices["date"])

    def safe(name: str) -> pd.DataFrame:
        try:
            return store.read(name)
        except FileNotFoundError:
            log.warning(f"Table '{name}' not found - returning empty frame")
            return pd.DataFrame()

    constituents = safe("constituents")
    if not constituents.empty:
        constituents["date"] = pd.to_datetime(constituents["date"])
    fundamentals = safe("fundamentals")
    if not fundamentals.empty and "report_date" in fundamentals.columns:
        fundamentals["report_date"] = pd.to_datetime(fundamentals["report_date"])

    benchmark = None
    try:
        bm = store.read("benchmark")
        if not bm.empty:
            bm["date"] = pd.to_datetime(bm["date"])
            benchmark = pd.Series(
                bm["value"].to_numpy(), index=pd.DatetimeIndex(bm["date"]), name="benchmark"
            ).sort_index()
    except FileNotFoundError:
        log.info("No persisted benchmark series found")

    return DataBundle(
        prices=prices,
        fundamentals=fundamentals,
        macro=safe("macro"),
        constituents=constituents,
        metadata={"source": str(root), "provenance": "PERSISTED"},
        benchmark=benchmark,
    )


def _fetch_benchmark_returns(
    provider: DataProvider, index_id: str, start: str, end: str
) -> pd.Series | None:
    """Fetch the provider benchmark and return it as a daily *return* series."""
    try:
        bench = provider.benchmark_prices(index_id, start, end)
    except NotImplementedError:
        log.info("Provider supplies no benchmark series")
        return None
    if bench is None or len(bench) == 0:
        return None
    bench = pd.Series(bench).sort_index()
    rets = bench.pct_change(fill_method=None).dropna()
    rets.name = "benchmark"
    return rets


__all__ = ["DataPipeline", "PipelineResult", "load_bundle"]

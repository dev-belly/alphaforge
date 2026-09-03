"""Data provider adapters behind a single :class:`DataProvider` interface."""

from alphaforge.data.providers.base import GICS_SECTORS, DataBundle, DataProvider
from alphaforge.data.providers.local import LocalParquetProvider
from alphaforge.data.providers.sample import PROVENANCE, SampleDataProvider, SampleSpec

__all__ = [
    "DataProvider",
    "DataBundle",
    "GICS_SECTORS",
    "SampleDataProvider",
    "SampleSpec",
    "LocalParquetProvider",
    "PROVENANCE",
]


def get_provider(name: str, **kwargs) -> DataProvider:
    """Factory used by the pipeline so providers stay swappable via config."""
    name = (name or "sample").lower()
    if name in {"sample", "synthetic"}:
        return SampleDataProvider(**kwargs)
    if name in {"local", "parquet"}:
        return LocalParquetProvider(**kwargs)
    if name == "yahoo":
        from alphaforge.data.providers.vendors import YahooFinanceProvider

        return YahooFinanceProvider(**kwargs)
    if name in {"akshare", "eastmoney"}:
        from alphaforge.data.providers.vendors import EastMoneyProvider

        return EastMoneyProvider(**kwargs)
    raise ValueError(f"Unknown data provider: {name!r}")

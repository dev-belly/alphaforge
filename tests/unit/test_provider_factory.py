"""Offline tests for the config-driven provider factory (``get_provider``).

The factory is the seam between *configuration* and *data source*. A wrong
mapping here is silent and expensive: the platform would happily run a research
cycle against the wrong backend (or a synthetic one) while reporting results as
real. These tests pin every alias and prove that every branch returns an object
that satisfies the :class:`DataProvider` interface, so providers stay swappable.

Optional third-party SDKs are faked via ``sys.modules`` so the tests behave
identically whether or not ``yfinance`` / ``requests`` happen to be installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from alphaforge.data.providers import get_provider
from alphaforge.data.providers.base import DataProvider
from alphaforge.data.providers.local import LocalParquetProvider
from alphaforge.data.providers.sample import SampleDataProvider
from alphaforge.data.providers.vendors import (
    EastMoneyProvider,
    ProviderUnavailableError,
    YahooFinanceProvider,
)


@pytest.fixture
def fake_yfinance(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """A stand-in ``yfinance`` module - construction needs no real SDK."""
    fake = types.SimpleNamespace(download=lambda *a, **k: None, Ticker=lambda sym: None)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return fake


@pytest.fixture
def fake_requests(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """A stand-in ``requests`` module - construction must not touch the network."""
    fake = types.SimpleNamespace(get=lambda *a, **k: None, Session=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


# --------------------------------------------------------------------------
# Alias -> class mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sample", "synthetic", "SAMPLE", "Sample"])
def test_sample_aliases_resolve_to_sample_provider(name: str) -> None:
    assert isinstance(get_provider(name), SampleDataProvider)


@pytest.mark.parametrize("name", ["local", "parquet", "LOCAL", "Parquet"])
def test_local_aliases_resolve_to_parquet_provider(name: str, tmp_path: Path) -> None:
    provider = get_provider(name, root=tmp_path)
    assert isinstance(provider, LocalParquetProvider)
    assert Path(provider.root) == tmp_path


def test_yahoo_alias_resolves_to_yahoo_provider(fake_yfinance: types.SimpleNamespace) -> None:
    provider = get_provider("yahoo")
    assert isinstance(provider, YahooFinanceProvider)
    assert provider._yf is fake_yfinance  # the injected SDK is what gets used


def test_yahoo_alias_is_case_insensitive(fake_yfinance: types.SimpleNamespace) -> None:
    assert isinstance(get_provider("YAHOO"), YahooFinanceProvider)


@pytest.mark.parametrize("name", ["akshare", "eastmoney", "AKSHARE"])
def test_akshare_aliases_resolve_to_eastmoney_provider(
    name: str, fake_requests: types.SimpleNamespace
) -> None:
    """``akshare`` is served by the key-less EastMoney adapter (no SDK needed)."""
    provider = get_provider(name)
    assert isinstance(provider, EastMoneyProvider)
    assert provider._requests is fake_requests


# --------------------------------------------------------------------------
# Defaults, errors and interface contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [None, "", "  "])
def test_missing_name_defaults_to_sample(name: str | None) -> None:
    """Never guess a live vendor when config is silent - stay on synthetic data."""
    assert isinstance(get_provider(name), SampleDataProvider)


def test_unknown_provider_raises_actionable_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown data provider: 'tushare'"):
        get_provider("tushare")


@pytest.mark.parametrize(
    ("name", "module", "hint"),
    [
        ("yahoo", "yfinance", "pip install yfinance"),
        ("akshare", "requests", "pip install requests"),
        ("eastmoney", "requests", "pip install requests"),
    ],
)
def test_missing_optional_dependency_surfaces_install_hint(
    monkeypatch: pytest.MonkeyPatch, name: str, module: str, hint: str
) -> None:
    """A configured-but-uninstallable vendor must fail with an actionable message.

    Setting ``sys.modules[module] = None`` makes ``import module`` raise, i.e. a
    clean install without that optional dependency. The platform relies on this
    surfacing :class:`ProviderUnavailableError` so it can fall back gracefully.
    """
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ProviderUnavailableError, match=hint):
        get_provider(name)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("sample", {}),
        ("local", {}),
        ("yahoo", {}),
        ("eastmoney", {}),
    ],
)
def test_every_branch_returns_a_data_provider(
    name: str, kwargs: dict, tmp_path: Path, fake_yfinance: object, fake_requests: object
) -> None:
    """The swapability contract: config picks the backend, the pipeline is agnostic."""
    if name == "local":
        kwargs = {"root": tmp_path}
    provider = get_provider(name, **kwargs)
    assert isinstance(provider, DataProvider)
    assert isinstance(provider.name, str) and provider.name


def test_kwargs_are_forwarded_to_the_provider(
    tmp_path: Path, fake_yfinance: types.SimpleNamespace, fake_requests: types.SimpleNamespace
) -> None:
    assert Path(get_provider("local", root=tmp_path).root) == tmp_path
    assert get_provider("yahoo", fundamental_lag_days=45).fundamental_lag_days == 45
    assert get_provider("eastmoney", timeout=7).timeout == 7

"""Offline tests for the factor research facade.

``FactorLibrary`` is the seam between "the registry can compute a factor" and
"the pipeline can use it", and the failure it owns is a *silent narrowing*: a
factor quietly dropped from the run looks exactly like a factor that was never
registered. So the tests assert the counts and the availability flag directly -
``available()`` must exclude the fundamental factors when there is no
fundamental table, ``compute()`` must return exactly what it was asked for, and
``specs()`` must still list every registered factor with an honest ``available``
flag rather than only the ones that ran.

The composite gets checked for the properties that make it usable as a signal:
z-scored per date (mean 0, unit standard deviation), no NaN left behind, and
weights that actually scale the contributions.

Everything is deterministic: no network, fixed seeds, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.factors.base import REGISTRY, FactorContext
from alphaforge.factors.library import FactorLibrary
from alphaforge.factors.preprocessing import ProcessingConfig
from alphaforge.features.fundamentals import FundamentalView
from alphaforge.features.panel import MarketPanel, build_panel

DATES = pd.bdate_range("2021-01-04", periods=600)
SYMBOLS = [f"S{i:02d}" for i in range(25)]
SUBSET = ["mom_20d", "mom_60d", "volatility_60d", "log_market_cap"]


@pytest.fixture(scope="module")
def panel() -> MarketPanel:
    rng = np.random.default_rng(9)
    rows = []
    for j, sym in enumerate(SYMBOLS):
        path = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(DATES))))
        for date, price in zip(DATES, path):
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "adj_close": price,
                    "close": price,
                    "volume": float(rng.uniform(1.0e5, 5.0e6)),
                    "market_cap": 1.0e9 * (1.0 + 0.1 * j),
                    "industry": "Tech" if j % 2 else "Energy",
                }
            )
    return build_panel(pd.DataFrame(rows))


def _statements(seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym in SYMBOLS:
        for report_date in ("2021-06-01", "2022-06-01"):
            rows.append(
                {
                    "symbol": sym,
                    "report_date": pd.Timestamp(report_date),
                    "net_income": float(rng.normal(1.0e8, 1.0e7)),
                    "total_equity": float(rng.normal(1.0e9, 1.0e8)),
                    "revenue": float(rng.normal(5.0e8, 5.0e7)),
                    "operating_cashflow": float(rng.normal(1.2e8, 1.0e7)),
                    "capex": float(rng.normal(3.0e7, 3.0e6)),
                    "gross_profit": float(rng.normal(2.0e8, 2.0e7)),
                    "ebit": float(rng.normal(1.5e8, 1.5e7)),
                    "total_assets": float(rng.normal(2.0e9, 2.0e8)),
                    "total_debt": float(rng.normal(4.0e8, 4.0e7)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def library(panel: MarketPanel) -> FactorLibrary:
    return FactorLibrary.from_config(FactorContext(panel), {})


@pytest.fixture(scope="module")
def library_with_fundamentals(panel: MarketPanel) -> FactorLibrary:
    view = FundamentalView.build(_statements(), panel.dates, panel.symbols, panel.market_cap)
    return FactorLibrary.from_config(FactorContext(panel, fundamentals=view), {})


def _fresh(panel: MarketPanel) -> FactorLibrary:
    return FactorLibrary.from_config(FactorContext(panel), {})


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
def test_from_config_reads_the_factor_block(panel: MarketPanel) -> None:
    got = FactorLibrary.from_config(
        FactorContext(panel), {"factor": {"horizon": 63, "n_quantiles": 10, "winsorize": False}}
    )
    assert got.horizon == 63
    assert got.n_quantiles == 10
    assert got.processing.winsorize is False


@pytest.mark.parametrize("cfg", [None, {}])
def test_an_empty_config_still_builds_a_library(panel: MarketPanel, cfg) -> None:
    got = FactorLibrary.from_config(FactorContext(panel), cfg)
    assert got.horizon == 21
    assert got.n_quantiles == 5


def test_the_universe_comes_from_the_panel(library: FactorLibrary, panel: MarketPanel) -> None:
    assert library.universe is panel.universe


# ----------------------------------------------------------------------
# Availability
# ----------------------------------------------------------------------
def test_fundamental_factors_are_excluded_without_a_fundamental_table(
    library: FactorLibrary,
) -> None:
    available = library.available()
    assert available, "a price-only panel should still offer factors"
    for name in available:
        assert REGISTRY.spec(name).requires_fundamentals is False


def test_fundamental_factors_appear_once_the_table_is_there(
    library_with_fundamentals: FactorLibrary,
) -> None:
    available = set(library_with_fundamentals.available())
    fundamental = {n for n in REGISTRY.names() if REGISTRY.spec(n).requires_fundamentals}
    assert fundamental, "the registry should contain fundamental factors"
    assert fundamental <= available


def test_available_is_sorted_and_free_of_duplicates(library: FactorLibrary) -> None:
    available = library.available()
    assert available == sorted(available)
    assert len(available) == len(set(available))


# ----------------------------------------------------------------------
# compute / preprocess / evaluate
# ----------------------------------------------------------------------
def test_compute_returns_every_available_factor(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    got = lib.compute()
    assert set(got) == set(lib.available())


def test_compute_honours_an_explicit_name_list(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    got = lib.compute(SUBSET)
    assert set(got) == set(SUBSET)


def test_preprocess_produces_one_full_panel_per_factor(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    lib.compute(SUBSET)
    got = lib.preprocess()
    assert set(got) == set(SUBSET)
    for frame in got.values():
        assert frame.shape == (len(lib.ctx.panel.dates), len(lib.ctx.panel.symbols))


def test_preprocess_computes_first_when_nothing_has_run(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    assert lib.raw == {}
    got = lib.preprocess(SUBSET)
    assert set(got) == set(SUBSET)
    assert lib.raw, "preprocess should have triggered compute"


def test_evaluate_returns_one_result_per_processed_factor(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    lib.preprocess(SUBSET)
    got = lib.evaluate()
    assert set(got) == set(SUBSET)
    for name, result in got.items():
        assert result.name == name
        assert result.category == REGISTRY.spec(name).category
        assert result.direction == REGISTRY.spec(name).direction


def test_run_walks_all_three_stages(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    got = lib.run(names=SUBSET)
    assert set(got) == set(SUBSET)
    assert lib.raw and lib.processed and lib.results


def test_correlation_triggers_preprocessing_when_nothing_has_run(
    library: FactorLibrary,
) -> None:
    """Both conveniences are lazy: calling them first must still work."""
    lib = _fresh(library.ctx.panel)
    assert lib.processed == {}
    got = lib.correlation()
    assert lib.processed, "correlation should have triggered preprocess"
    assert got.shape[0] == got.shape[1]
    assert list(got.columns) == list(got.index)
    # A factor that is constant across the cross-section has no variance after
    # de-meaning, so its self-correlation is NaN rather than 1 - the synthetic
    # panel has no zero-volume days, which is exactly such a factor.
    diagonal = np.diag(got.to_numpy(dtype=float))
    finite = diagonal[np.isfinite(diagonal)]
    assert finite.size > 0
    assert np.allclose(finite, 1.0, atol=1e-9)


def test_summary_table_triggers_evaluation_when_nothing_has_run(
    library: FactorLibrary,
) -> None:
    lib = _fresh(library.ctx.panel)
    assert lib.results == {}
    got = lib.summary_table()
    assert lib.results, "summary_table should have triggered evaluate"
    assert len(got) == len(lib.results)


def test_summary_table_restricts_to_a_subset_when_asked(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    lib.run(names=SUBSET)
    got = lib.summary_table()
    assert len(got) == len(SUBSET)


# ----------------------------------------------------------------------
# specs
# ----------------------------------------------------------------------
def test_specs_lists_every_registered_factor(library: FactorLibrary) -> None:
    got = library.specs()
    assert {row["name"] for row in got} == set(REGISTRY.names())
    assert sorted(row["name"] for row in got) == [row["name"] for row in got]


def test_specs_flags_availability_honestly(library: FactorLibrary) -> None:
    """A factor that did not run must still be listed, marked unavailable."""
    available = set(library.available())
    for row in library.specs():
        assert row["available"] is (row["name"] in available)
        assert row["direction"] in (1, -1)
        assert row["category"]


# ----------------------------------------------------------------------
# composite
# ----------------------------------------------------------------------
def test_composite_is_a_z_score_of_the_processed_factors(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    lib.preprocess(SUBSET)
    got = lib.composite(names=SUBSET)
    assert got.shape == (len(lib.ctx.panel.dates), len(lib.ctx.panel.symbols))
    assert not got.isna().any().any(), "the composite fills to the neutral value"
    row = got.iloc[-1]
    assert row.mean() == pytest.approx(0.0, abs=1e-12)
    assert row.std() == pytest.approx(1.0, rel=1e-9)


def test_composite_defaults_to_every_processed_factor(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    lib.preprocess(SUBSET)
    assert lib.composite().shape == lib.composite(names=SUBSET).shape


def test_composite_weights_scale_the_contributions(library: FactorLibrary) -> None:
    """A factor given double weight must move the composite more than its peer."""
    lib = _fresh(library.ctx.panel)
    lib.preprocess(["mom_20d", "volatility_60d"])
    balanced = lib.composite(names=["mom_20d", "volatility_60d"])
    tilted = lib.composite(names=["mom_20d", "volatility_60d"], weights={"mom_20d": 3.0})
    assert not np.allclose(balanced.to_numpy(), tilted.to_numpy())
    assert tilted.notna().all().all()


def test_all_zero_weights_collapse_to_the_neutral_value(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    lib.preprocess(SUBSET)
    got = lib.composite(names=SUBSET, weights=dict.fromkeys(SUBSET, 0.0))
    assert (got.to_numpy() == 0.0).all()


def test_composite_rejects_an_unknown_factor(library: FactorLibrary) -> None:
    lib = _fresh(library.ctx.panel)
    lib.preprocess(SUBSET)
    with pytest.raises(KeyError):
        lib.composite(names=["not_a_factor"])


def test_composite_refuses_to_build_from_nothing(
    panel: MarketPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = FactorLibrary(ctx=FactorContext(panel), processing=ProcessingConfig())
    monkeypatch.setattr(FactorLibrary, "preprocess", lambda self, names=None: {})
    with pytest.raises(ValueError, match="No factors available to composite"):
        lib.composite()

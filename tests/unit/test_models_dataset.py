"""Offline tests for the supervised dataset builder.

Two things here are silently catastrophic when they go wrong:

**Alignment.**  The long frame is built by ``ravel()``-ing each ``(dates x
symbols)`` panel into a pre-sized ``MultiIndex.from_product([dates, symbols])``.
If those two orders ever disagree, every feature is attached to the wrong
(date, symbol) and the model trains on shuffled noise - with no error, no NaN
and no shape mismatch. So the tests below assert a feature value *by position*:
a factor panel encoded as ``t * 100 + s`` must come back as ``t * 100 + s``.

**The label.**  ``forward_return`` is ``close[t + horizon] / close[t] - 1``, so
it is only knowable after the holding period. The tests recompute it from the
panel and merge on ``(date, symbol)`` rather than trusting the builder.

The two ranked target modes are also pinned to their documented ranges:
``forward_rank`` in ``(0, 1]`` and ``forward_return`` in ``(-0.5, 0.5]``. Both
used to emit the centred version, which made ``target`` a no-op parameter.

Everything is deterministic: no network, fixed seeds, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.features.panel import MarketPanel, build_panel
from alphaforge.models.dataset import AlphaDataset, build_dataset, to_matrix

DATES = pd.bdate_range("2024-01-02", periods=12)
SYMBOLS = [f"S{i}" for i in range(8)]
HORIZON = 3
MIN_NAMES = 5


def _panel(seed: int = 4) -> MarketPanel:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.02, size=(len(DATES), len(SYMBOLS)))
    prices = 100.0 * np.exp(np.cumsum(steps, axis=0))
    rows = []
    for s_idx, sym in enumerate(SYMBOLS):
        for t, date in enumerate(DATES):
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "adj_close": prices[t, s_idx],
                    "close": prices[t, s_idx],
                    "volume": 1.0e6,
                    "market_cap": 1.0e9,
                    "industry": "Tech" if s_idx % 2 == 0 else "Energy",
                }
            )
    return build_panel(pd.DataFrame(rows))


def _encoded_panels(panel: MarketPanel) -> dict[str, pd.DataFrame]:
    """A factor panel whose value encodes its own (date, symbol) position."""
    t_idx = np.arange(len(DATES))
    s_idx = np.arange(len(SYMBOLS))
    encoded = t_idx[:, None] * 100.0 + s_idx[None, :]
    grid = pd.DataFrame(encoded, index=DATES, columns=SYMBOLS)
    return {
        "position": grid,
        "negative": -grid,
        "constant": pd.DataFrame(1.0, index=DATES, columns=SYMBOLS),
    }


def _expected_forward(panel: MarketPanel, horizon: int = HORIZON) -> pd.DataFrame:
    fwd = (panel.close.shift(-horizon) / panel.close - 1.0).where(panel.universe)
    ref = fwd.stack().rename("ref").reset_index()
    return ref.rename(columns={"level_0": "date", "level_1": "symbol"})


@pytest.fixture
def panel() -> MarketPanel:
    return _panel()


@pytest.fixture
def dataset(panel: MarketPanel) -> AlphaDataset:
    return build_dataset(
        panel, _encoded_panels(panel), horizon=HORIZON, min_names_per_date=MIN_NAMES
    )


def _merged_with_reference(ds: AlphaDataset, panel: MarketPanel) -> pd.DataFrame:
    got = pd.DataFrame(
        {"date": ds.dates, "symbol": ds.symbols, "fwd": ds.forward_returns.to_numpy()}
    )
    merged = got.merge(_expected_forward(panel), on=["date", "symbol"], how="left")
    assert len(merged) == len(got), "the reference has duplicate (date, symbol) rows"
    return merged


# ----------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------
def test_a_feature_value_lands_on_its_own_date_and_symbol(dataset: AlphaDataset) -> None:
    """The ravel order and the MultiIndex product order must agree."""
    t_map = {d: t for t, d in enumerate(DATES)}
    s_map = {s: i for i, s in enumerate(SYMBOLS)}
    expected = dataset.dates.map(t_map) * 100.0 + dataset.symbols.map(s_map)
    np.testing.assert_allclose(dataset.features["position"].to_numpy(), expected.to_numpy())
    np.testing.assert_allclose(dataset.features["negative"].to_numpy(), -expected.to_numpy())


def test_the_label_matches_an_independently_computed_forward_return(
    dataset: AlphaDataset, panel: MarketPanel
) -> None:
    merged = _merged_with_reference(dataset, panel)
    np.testing.assert_allclose(
        merged["fwd"].to_numpy(dtype=float),
        merged["ref"].to_numpy(dtype=float),
        rtol=1e-12,
        equal_nan=True,
    )


def test_the_label_is_not_the_contemporaneous_return(
    dataset: AlphaDataset, panel: MarketPanel
) -> None:
    """A backward-looking label would make every backtest a fantasy."""
    merged = _merged_with_reference(dataset, panel)
    same_day = (panel.close / panel.close.shift(1) - 1.0).stack().rename("same_day").reset_index()
    same_day = same_day.rename(columns={"level_0": "date", "level_1": "symbol"})
    joined = merged.merge(same_day, on=["date", "symbol"], how="left")
    difference = (joined["fwd"] - joined["same_day"]).abs().dropna()
    assert len(difference) > 0
    assert (difference > 1e-9).all(), "the label looks like the same-day return"


def test_the_feature_order_follows_the_dict_order(dataset: AlphaDataset) -> None:
    assert dataset.feature_names == ["position", "negative", "constant"]


def test_the_long_frame_covers_every_date_and_symbol_pair(panel: MarketPanel) -> None:
    """Before any filtering the grid is complete: one row per (date, symbol)."""
    ds = build_dataset(
        panel, _encoded_panels(panel), horizon=HORIZON, min_names_per_date=1, dropna_labels=False
    )
    assert len(ds) == len(DATES) * len(SYMBOLS)
    assert ds.dates.nunique() == len(DATES)
    assert ds.symbols.nunique() == len(SYMBOLS)


# ----------------------------------------------------------------------
# Target modes
# ----------------------------------------------------------------------
def test_forward_rank_is_bounded_in_zero_to_one(dataset: AlphaDataset) -> None:
    """Regression: both ranked modes used to emit the centred version."""
    target = dataset.target
    assert dataset.metadata["target_type"] == "forward_rank"
    assert target.min() > 0.0
    assert target.max() == pytest.approx(1.0)


def test_forward_return_is_bounded_in_minus_half_to_half(panel: MarketPanel) -> None:
    ds = build_dataset(
        panel,
        _encoded_panels(panel),
        horizon=HORIZON,
        target="forward_return",
        min_names_per_date=MIN_NAMES,
    )
    assert ds.target.min() > -0.5
    assert ds.target.max() == pytest.approx(0.5)
    assert ds.metadata["target_type"] == "forward_return"


def test_the_two_ranked_modes_differ_by_exactly_one_half(panel: MarketPanel) -> None:
    panels = _encoded_panels(panel)
    rank = build_dataset(
        panel, panels, horizon=HORIZON, target="forward_rank", min_names_per_date=MIN_NAMES
    )
    centred = build_dataset(
        panel, panels, horizon=HORIZON, target="forward_return", min_names_per_date=MIN_NAMES
    )
    same_rows = (rank.dates == centred.dates).to_numpy() & (
        rank.symbols == centred.symbols
    ).to_numpy()
    assert same_rows.all(), "the two modes must select identical rows"
    np.testing.assert_allclose((rank.target - centred.target).to_numpy(), np.full(len(rank), 0.5))


def test_an_unknown_target_mode_keeps_the_raw_return(panel: MarketPanel) -> None:
    ds = build_dataset(
        panel,
        _encoded_panels(panel),
        horizon=HORIZON,
        target="raw_diagnostic",
        min_names_per_date=MIN_NAMES,
    )
    np.testing.assert_allclose(
        ds.target.to_numpy(dtype=float), ds.forward_returns.to_numpy(dtype=float)
    )


def test_the_target_is_a_cross_sectional_rank_not_a_global_one(dataset: AlphaDataset) -> None:
    """Ranking globally would leak the market's time-series drift into the label."""
    per_date = dataset.target.groupby(dataset.dates).max()
    np.testing.assert_allclose(per_date.to_numpy(), np.full(len(per_date), 1.0))
    assert dataset.target.groupby(dataset.dates).size().nunique() == 1


def test_the_target_is_nan_wherever_the_label_is(dataset: AlphaDataset) -> None:
    missing = dataset.forward_returns.isna()
    assert dataset.target[missing].isna().all()


def test_a_horizon_longer_than_the_sample_produces_no_label(panel: MarketPanel) -> None:
    ds = build_dataset(panel, _encoded_panels(panel), horizon=10_000, min_names_per_date=MIN_NAMES)
    assert len(ds) == 0
    assert ds.describe()["n_samples"] == 0


# ----------------------------------------------------------------------
# Missing data
# ----------------------------------------------------------------------
def test_a_missing_feature_is_filled_with_a_neutral_zero(panel: MarketPanel) -> None:
    panels = _encoded_panels(panel)
    broken = panels["negative"].copy()
    broken.iloc[2, 3] = np.nan
    panels["negative"] = broken
    ds = build_dataset(panel, panels, horizon=HORIZON, min_names_per_date=MIN_NAMES)
    assert not ds.features.isna().any().any()
    patched = ds.features[(ds.dates == DATES[2]) & (ds.symbols == SYMBOLS[3])]
    assert patched["negative"].iloc[0] == 0.0
    # Everything else keeps its real value.
    untouched = ds.features[(ds.dates == DATES[2]) & (ds.symbols == SYMBOLS[4])]
    assert untouched["negative"].iloc[0] != 0.0


def test_a_name_outside_the_universe_is_neutralised_and_unlabelled(panel: MarketPanel) -> None:
    panel.universe.loc[DATES[4], "S0"] = False
    ds = build_dataset(panel, _encoded_panels(panel), horizon=HORIZON, min_names_per_date=MIN_NAMES)
    row = ds.features[(ds.dates == DATES[4]) & (ds.symbols == "S0")]
    assert row.empty, "an untradable name must not survive the label drop"
    kept = ds.features[(ds.dates == DATES[4]) & (ds.symbols == "S1")]
    assert not kept.empty


def test_dates_below_the_minimum_breadth_are_dropped(panel: MarketPanel) -> None:
    assert (
        len(
            build_dataset(
                panel, _encoded_panels(panel), horizon=HORIZON, min_names_per_date=len(SYMBOLS) + 1
            )
        )
        == 0
    )


def test_dropna_labels_false_keeps_the_unlabelled_tail(panel: MarketPanel) -> None:
    panels = _encoded_panels(panel)
    kept = build_dataset(panel, panels, horizon=HORIZON, min_names_per_date=MIN_NAMES)
    allrows = build_dataset(
        panel, panels, horizon=HORIZON, min_names_per_date=MIN_NAMES, dropna_labels=False
    )
    assert len(allrows) > len(kept)
    assert allrows.target.isna().any(), "the end-of-sample rows should carry no label"
    assert not allrows.features.isna().any().any()


def test_dropna_labels_false_still_drops_dates_that_shrink_below_the_floor(
    panel: MarketPanel,
) -> None:
    """A date left with too few *labelled* names must not survive on feature count alone."""
    panels = _encoded_panels(panel)
    min_names = len(SYMBOLS) - 2
    allrows = build_dataset(
        panel, panels, horizon=HORIZON, min_names_per_date=min_names, dropna_labels=False
    )
    survivors = allrows.dates.value_counts()
    assert len(survivors) > 0
    assert (survivors >= min_names).all()


# ----------------------------------------------------------------------
# AlphaDataset helpers
# ----------------------------------------------------------------------
def test_len_is_the_number_of_samples(dataset: AlphaDataset) -> None:
    assert len(dataset) == len(dataset.target) == len(dataset.features)


def test_date_index_is_sorted_and_unique(dataset: AlphaDataset) -> None:
    got = dataset.date_index()
    assert isinstance(got, pd.DatetimeIndex)
    assert got.is_monotonic_increasing
    assert got.is_unique
    assert set(got) <= set(DATES)


def test_mask_by_dates_selects_the_right_rows(dataset: AlphaDataset) -> None:
    mask = dataset.mask_by_dates(pd.DatetimeIndex([DATES[0]]))
    assert mask.dtype == bool
    assert mask.sum() == int((dataset.dates == DATES[0]).sum())
    assert (dataset.dates[mask] == DATES[0]).all()


def test_describe_reports_the_shape_and_the_span(dataset: AlphaDataset) -> None:
    got = dataset.describe()
    assert got["n_samples"] == len(dataset)
    assert got["n_features"] == len(dataset.feature_names)
    assert got["n_dates"] == dataset.dates.nunique()
    assert got["n_symbols"] == dataset.symbols.nunique()
    assert got["target"] == "forward_rank"
    assert got["start"] < got["end"]


def test_describe_survives_an_empty_dataset(panel: MarketPanel) -> None:
    ds = build_dataset(panel, _encoded_panels(panel), horizon=10_000, min_names_per_date=MIN_NAMES)
    got = ds.describe()
    assert got["n_samples"] == 0
    assert got["n_dates"] == 0


# ----------------------------------------------------------------------
# to_matrix
# ----------------------------------------------------------------------
def test_to_matrix_returns_arrays_and_a_meta_frame(dataset: AlphaDataset) -> None:
    X, y, meta = to_matrix(dataset)
    assert X.shape == (len(dataset), len(dataset.feature_names))
    assert X.dtype == float
    assert y.dtype == float
    assert list(meta.columns) == ["date", "symbol"]
    assert len(meta) == len(dataset)


def test_to_matrix_honours_a_mask(dataset: AlphaDataset) -> None:
    mask = dataset.mask_by_dates(pd.DatetimeIndex([DATES[0], DATES[1]]))
    X, y, meta = to_matrix(dataset, mask)
    assert X.shape == (int(mask.sum()), len(dataset.feature_names))
    assert len(y) == int(mask.sum())
    assert set(meta["date"]) == {DATES[0], DATES[1]}


def test_to_matrix_preserves_the_row_order(dataset: AlphaDataset) -> None:
    X, _, meta = to_matrix(dataset)
    np.testing.assert_allclose(X[:, 0], dataset.features["position"].to_numpy())
    assert (meta["symbol"].to_numpy() == dataset.symbols.to_numpy()).all()


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
def test_no_factor_panels_is_rejected(panel: MarketPanel) -> None:
    with pytest.raises(ValueError, match="No factor panels supplied"):
        build_dataset(panel, {}, horizon=HORIZON)


def test_an_explicit_universe_overrides_the_panel(panel: MarketPanel) -> None:
    narrow = pd.DataFrame(False, index=panel.dates, columns=panel.symbols)
    narrow.loc[:, "S0"] = True
    ds = build_dataset(
        panel,
        _encoded_panels(panel),
        horizon=HORIZON,
        universe=narrow,
        min_names_per_date=1,
    )
    assert set(ds.symbols) == {"S0"}

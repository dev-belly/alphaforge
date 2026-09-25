"""Offline tests for the investable-universe builder.

The universe decides which names a backtest is *allowed* to trade, so every way
of getting it wrong is a way of getting a flattering result. The two that matter
most:

  * **membership must not leak backwards.** Membership is forward-filled from the
    snapshot date, never backward - learning that a stock joined the index last
    month must not make it investable last month. The tests assert that a date
    before the first snapshot is empty and that a snapshot only takes effect from
    its own date onward.
  * **the screens must be trailing-only.** Price, ADV and history are all rolling
    windows over past data; a tampered future must not move an earlier day's
    eligibility.

Also pinned: a fresh listing is not investable on day one (the ``min_history``
screen), the ``max_names`` cap keeps the largest capitalisations, and the
survivorship diagnostic counts a name as delisted only after a real gap.

Everything is deterministic: no network, no provider, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.data.universe import Universe, UniverseConfig

DATES = pd.bdate_range("2022-01-03", periods=200)
SYMBOLS = [f"S{i:02d}" for i in range(6)]


def _prices(
    *,
    symbols: list[str] | None = None,
    price: float = 50.0,
    volume: float = 1.0e6,
    market_cap: float = 1.0e9,
) -> pd.DataFrame:
    symbols = symbols or SYMBOLS
    rows = []
    for j, sym in enumerate(symbols):
        for date in DATES:
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "adj_close": price * (1.0 + 0.01 * j),
                    "volume": volume,
                    "market_cap": market_cap * (1.0 + j),
                }
            )
    return pd.DataFrame(rows)


def _constituents(snapshots: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for date, members in snapshots.items():
        for sym in members:
            rows.append({"date": pd.Timestamp(date), "symbol": sym})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# membership_matrix
# ----------------------------------------------------------------------
def test_no_constituents_means_everything_is_a_member() -> None:
    """A provider without membership data must not silently empty the universe."""
    got = Universe().membership_matrix(None, DATES, pd.Index(SYMBOLS))
    assert got.shape == (len(DATES), len(SYMBOLS))
    assert got.to_numpy().all()


def test_an_empty_constituent_frame_is_the_same_as_none() -> None:
    got = Universe().membership_matrix(pd.DataFrame(), DATES, pd.Index(SYMBOLS))
    assert got.to_numpy().all()


def test_membership_before_the_first_snapshot_is_empty() -> None:
    """Nothing is known before the first snapshot, so nothing is investable."""
    cons = _constituents({"2022-06-01": ["S00", "S01"]})
    got = Universe().membership_matrix(cons, DATES, pd.Index(SYMBOLS))
    assert not got.loc[got.index < pd.Timestamp("2022-06-01")].any().any()


def test_membership_is_forward_filled_from_the_snapshot() -> None:
    cons = _constituents({"2022-06-01": ["S00"]})
    got = Universe().membership_matrix(cons, DATES, pd.Index(SYMBOLS))
    after = got.loc[got.index >= pd.Timestamp("2022-06-01")]
    assert after["S00"].all()
    assert not after.drop(columns=["S00"]).any().any()


def test_a_later_snapshot_replaces_the_earlier_one() -> None:
    cons = _constituents({"2022-06-01": ["S00"], "2022-09-01": ["S01"]})
    got = Universe().membership_matrix(cons, DATES, pd.Index(SYMBOLS))
    assert got.loc[pd.Timestamp("2022-07-01"), "S00"]
    assert not got.loc[pd.Timestamp("2022-07-01"), "S01"]
    assert got.loc[pd.Timestamp("2022-10-03"), "S01"]
    assert not got.loc[pd.Timestamp("2022-10-03"), "S00"]


def test_a_snapshot_takes_effect_on_its_own_date() -> None:
    cons = _constituents({"2022-06-01": ["S00"]})
    got = Universe().membership_matrix(cons, DATES, pd.Index(SYMBOLS))
    snapshot_day = DATES[DATES.searchsorted(pd.Timestamp("2022-06-01"))]
    assert got.loc[snapshot_day, "S00"]


def test_symbols_outside_the_index_are_ignored() -> None:
    cons = _constituents({"2022-06-01": ["S00", "NOT_IN_PANEL"]})
    got = Universe().membership_matrix(cons, DATES, pd.Index(SYMBOLS))
    assert got.shape == (len(DATES), len(SYMBOLS))
    assert got.loc[pd.Timestamp("2022-07-01"), "S00"]


# ----------------------------------------------------------------------
# eligibility_mask
# ----------------------------------------------------------------------
def test_a_cheap_name_fails_the_price_screen() -> None:
    prices = _prices(price=2.0)
    got = Universe(UniverseConfig(min_price=5.0)).eligibility_mask(prices, DATES, pd.Index(SYMBOLS))
    assert not got.any().any()


def test_a_thinly_traded_name_fails_the_adv_screen() -> None:
    prices = _prices(price=50.0, volume=1.0)  # $50/day
    got = Universe(UniverseConfig(min_adv_usd=1.0e6)).eligibility_mask(
        prices, DATES, pd.Index(SYMBOLS)
    )
    assert not got.any().any()


def test_a_fresh_listing_is_not_investable_on_day_one() -> None:
    """The history screen is what stops a new issue being traded immediately."""
    prices = _prices(symbols=["S00"])
    prices.loc[prices.index[:100], "adj_close"] = np.nan  # first print at row 100
    got = Universe(UniverseConfig(min_history=60)).eligibility_mask(
        prices, DATES, pd.Index(["S00"])
    )
    assert not got["S00"].iloc[100], "day one of trading has no history"
    assert not got["S00"].iloc[158], "59 observations is still short of 60"
    assert got["S00"].iloc[159], "the 60th observation makes it investable"
    assert got["S00"].iloc[-1]


def test_a_fully_screened_name_is_eligible_once_it_has_history() -> None:
    """The history screen legitimately excludes the first ``min_history - 1`` days."""
    got = Universe(UniverseConfig(min_history=60)).eligibility_mask(
        _prices(), DATES, pd.Index(SYMBOLS)
    )
    assert not got.iloc[:59].any().any()
    assert got.iloc[59:].to_numpy().all()


def test_eligibility_never_sees_the_future() -> None:
    baseline = _prices()
    tampered = _prices()
    # Mask on the date column: the frame has one row per (date, symbol), so a
    # row-number mask would tamper with whole symbols rather than whole days.
    tampered.loc[tampered["date"] > DATES[150], "adj_close"] = 1.0  # crash late on
    cutoff = DATES[120]

    a = Universe().eligibility_mask(tampered, DATES, pd.Index(SYMBOLS))
    b = Universe().eligibility_mask(baseline, DATES, pd.Index(SYMBOLS))
    pd.testing.assert_frame_equal(a.loc[:cutoff], b.loc[:cutoff])


def test_eligibility_is_indexed_like_the_requested_grid() -> None:
    subset = pd.Index(SYMBOLS[:3])
    got = Universe().eligibility_mask(_prices(), DATES, subset)
    assert list(got.columns) == list(subset)
    assert list(got.index) == list(DATES)
    assert got.dtypes.eq(bool).all()


# ----------------------------------------------------------------------
# build
# ----------------------------------------------------------------------
def test_an_empty_price_panel_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty price panel"):
        Universe().build(pd.DataFrame(), None)


def test_build_intersects_membership_with_eligibility() -> None:
    cons = _constituents({"2022-01-03": ["S00", "S01"]})
    got = Universe().build(_prices(), cons)
    assert got["S00"].any() and got["S01"].any()
    assert not got.drop(columns=["S00", "S01"]).any().any()


def test_build_defaults_to_the_panel_dates() -> None:
    got = Universe().build(_prices(), None)
    assert list(got.index) == list(DATES)


def test_max_names_keeps_the_largest_capitalisations() -> None:
    """Market cap rises with the symbol index in the fixture, so S05 is biggest."""
    got = Universe(UniverseConfig(max_names=2)).build(_prices(), None)
    last = got.iloc[-1]
    assert last.sum() == 2
    assert last["S05"] and last["S04"]
    assert not last[["S00", "S01", "S02", "S03"]].any()


def test_max_names_is_applied_per_date() -> None:
    got = Universe(UniverseConfig(max_names=3)).build(_prices(), None)
    assert (got.sum(axis=1) <= 3).all()
    # Once the history screen is satisfied the cap binds exactly.
    assert (got.iloc[59:].sum(axis=1) == 3).all()


def test_max_names_none_keeps_every_eligible_name() -> None:
    got = Universe(UniverseConfig(max_names=None)).build(_prices(), None)
    assert got.iloc[-1].sum() == len(SYMBOLS)


def test_max_names_survives_a_date_with_nothing_eligible() -> None:
    """A date with no candidates must stay empty, not crash on an empty sort."""
    prices = _prices()
    prices.loc[prices["date"] > DATES[100], "adj_close"] = 1.0  # below min_price
    got = Universe(UniverseConfig(max_names=2)).build(prices, None)
    assert not got.iloc[-1].any()
    assert got.iloc[80].sum() == 2, "the cap still binds before the screen fails"


def test_build_returns_a_boolean_panel() -> None:
    got = Universe().build(_prices(), None)
    assert got.dtypes.eq(bool).all()
    assert list(got.columns) == SYMBOLS


# ----------------------------------------------------------------------
# survivorship_diagnostics
# ----------------------------------------------------------------------
def test_the_diagnostic_counts_names_that_stop_trading() -> None:
    prices = _prices()
    gone = prices["symbol"] == "S05"
    prices.loc[gone & (prices["date"] > DATES[100]), "adj_close"] = np.nan
    universe = Universe().build(prices, None)

    got = Universe.survivorship_diagnostics(prices, universe)
    assert got["n_symbols_total"] == len(SYMBOLS)
    assert got["n_delisted"] == 1
    assert got["delisted_pct"] == pytest.approx(1 / len(SYMBOLS))
    assert "terminal return" in got["note"]


def test_a_brief_gap_does_not_count_as_delisting() -> None:
    """The 30-day threshold tolerates a suspension, not just a missing session."""
    prices = _prices()
    prices.loc[(prices["symbol"] == "S05") & (prices["date"] == DATES[100]), "adj_close"] = np.nan
    got = Universe.survivorship_diagnostics(prices, Universe().build(prices, None))
    assert got["n_delisted"] == 0


def test_the_diagnostic_reports_how_many_names_were_investable() -> None:
    cons = _constituents({"2022-01-03": ["S00", "S01", "S02"]})
    prices = _prices()
    universe = Universe().build(prices, cons)
    got = Universe.survivorship_diagnostics(prices, universe)
    assert got["n_symbols_investable"] == 3
    assert got["n_symbols_total"] == len(SYMBOLS)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
def test_the_default_screens_are_the_documented_ones() -> None:
    cfg = UniverseConfig()
    assert cfg.min_price == 5.0
    assert cfg.min_adv_usd == 1_000_000.0
    assert cfg.min_history == 60
    assert cfg.max_names is None


def test_the_universe_defaults_to_a_default_config() -> None:
    assert Universe().config == UniverseConfig()
    custom = UniverseConfig(min_price=1.0)
    assert Universe(custom).config is custom

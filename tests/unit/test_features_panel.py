"""Offline tests for the wide market panel.

Every factor, model, optimiser and backtest consumes the :class:`MarketPanel`
built here, so a defect is not a local bug - it is silently wrong alpha, wrong
neutralisation and wrong risk for the whole research stack. These tests lock the
things that are easy to get quietly wrong:

  * returns come from **adjusted** prices and a price gap is never filled;
  * a source column that is missing yields an all-NaN **float** panel, not an
    all-NaN object panel that breaks every downstream comparison;
  * a substituted market cap is *labelled* (``market_cap_source``) and, when the
    substitution is itself impossible, admitted as unavailable;
  * the universe is reindexed onto the panel **and** masked by having a price;
  * the forward-return label is aligned to the signal date and only includes an
    execution lag when one is asked for.

Everything is deterministic: no network, no randomness, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.features import panel as panel_module
from alphaforge.features.panel import build_panel, industry_dummies

DATES = pd.bdate_range("2024-01-02", periods=300)
SYMBOLS = ["AAA", "BBB", "CCC"]
INDUSTRY = {"AAA": "Tech", "BBB": "Energy", "CCC": "Tech"}
HORIZON = 21


def _long_table() -> pd.DataFrame:
    """A clean long (date, symbol) price table with every column present."""
    frames = []
    for i, sym in enumerate(SYMBOLS):
        frames.append(
            pd.DataFrame(
                {
                    "date": DATES,
                    "symbol": sym,
                    "open": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "close": 100.0 + i + np.arange(len(DATES), dtype=float),
                    "adj_close": 10.0 + i + np.arange(len(DATES), dtype=float) * 0.1,
                    "volume": 1_000.0 * (i + 1) + np.arange(len(DATES), dtype=float),
                    "market_cap": 1.0e9 * (i + 1),
                    "industry": INDUSTRY[sym],
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out.attrs["provenance"] = "TEST_FIXTURE"
    return out


def _warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture ``log.warning`` - the project logger does not propagate to caplog."""
    seen: list[str] = []
    monkeypatch.setattr(panel_module.log, "warning", lambda msg, *a, **k: seen.append(str(msg)))
    return seen


# ----------------------------------------------------------------------
# Hard failures
# ----------------------------------------------------------------------
def test_an_empty_price_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty price frame"):
        build_panel(pd.DataFrame())


def test_a_frame_without_adjusted_prices_is_rejected() -> None:
    # Without adj_close every panel is NaN and the universe silently empties;
    # that must be a loud failure, not a zero-breadth panel.
    long = _long_table().drop(columns=["adj_close"])
    with pytest.raises(ValueError, match="adj_close"):
        build_panel(long)


# ----------------------------------------------------------------------
# Shape, alignment and prices
# ----------------------------------------------------------------------
def test_panel_is_dates_by_symbols_with_sorted_axes() -> None:
    panel = build_panel(_long_table())
    assert list(panel.dates) == list(DATES)
    assert panel.symbols.tolist() == SYMBOLS
    assert len(panel) == len(DATES)
    for frame in (panel.close, panel.raw_close, panel.volume, panel.market_cap):
        assert list(frame.index) == list(DATES)
        assert frame.columns.tolist() == SYMBOLS


def test_returns_are_computed_from_adjusted_prices() -> None:
    panel = build_panel(_long_table())
    pd.testing.assert_frame_equal(panel.returns, panel.close.pct_change(fill_method=None))
    # The raw close carries the corporate actions, so it must not be the source.
    assert not panel.returns.equals(panel.raw_close.pct_change(fill_method=None))


def test_a_price_gap_is_never_forward_filled() -> None:
    long = _long_table()
    gap = (long["symbol"] == "AAA") & (long["date"] == DATES[10])
    long.loc[gap, "adj_close"] = np.nan

    panel = build_panel(long)
    assert pd.isna(panel.close.loc[DATES[10], "AAA"])
    # No ffill: the missing price also removes the return that would have been
    # measured against it, rather than inventing a flat day.
    assert pd.isna(panel.returns.loc[DATES[10], "AAA"])
    assert pd.isna(panel.returns.loc[DATES[11], "AAA"])
    # ...and the day after the gap is the true 11 -> 12 move, not a move measured
    # across the hole.
    assert panel.returns.loc[DATES[12], "AAA"] == pytest.approx(
        panel.close.loc[DATES[12], "AAA"] / panel.close.loc[DATES[11], "AAA"] - 1.0
    )


def test_dollar_volume_is_price_times_volume() -> None:
    panel = build_panel(_long_table())
    pd.testing.assert_frame_equal(panel.dollar_volume, panel.close * panel.volume)


def test_a_missing_source_column_stays_float() -> None:
    # Regression: the missing-column branch used to build an object-dtype frame,
    # which makes `dollar_volume > 1e6` raise and `.rolling(252).median()` refuse
    # to aggregate.
    panel = build_panel(_long_table().drop(columns=["volume"]))
    assert panel.volume.dtypes.eq(np.dtype("float64")).all()
    assert panel.volume.isna().all().all()
    assert panel.dollar_volume.dtypes.eq(np.dtype("float64")).all()


# ----------------------------------------------------------------------
# Market capitalisation
# ----------------------------------------------------------------------
def test_a_reported_market_cap_is_used_as_is() -> None:
    panel = build_panel(_long_table())
    assert panel.metadata["market_cap_source"] == "reported"
    assert panel.market_cap.loc[DATES[0], "AAA"] == pytest.approx(1.0e9)
    assert panel.market_cap.loc[DATES[0], "CCC"] == pytest.approx(3.0e9)


def test_a_missing_market_cap_falls_back_to_a_labelled_proxy(monkeypatch) -> None:
    seen = _warnings(monkeypatch)
    long = _long_table()
    long["market_cap"] = pd.NA

    panel = build_panel(long)
    assert panel.metadata["market_cap_source"] == "dollar_volume_proxy"
    expected = panel.dollar_volume.rolling(252, min_periods=60).median()
    pd.testing.assert_frame_equal(panel.market_cap, expected)
    # The substitution is disclosed rather than presented as a real cap.
    assert any("dollar-volume proxy" in msg for msg in seen)


def test_market_cap_is_admitted_unavailable_when_the_proxy_is_empty(monkeypatch) -> None:
    seen = _warnings(monkeypatch)
    long = _long_table().drop(columns=["volume"])
    long["market_cap"] = pd.NA

    panel = build_panel(long)
    assert panel.metadata["market_cap_source"] == "unavailable"
    assert panel.market_cap.isna().all().all()
    assert any("unavailable" in msg or "no usable" in msg.lower() for msg in seen)


# ----------------------------------------------------------------------
# Universe and industry
# ----------------------------------------------------------------------
def test_universe_defaults_to_having_a_price() -> None:
    long = _long_table()
    long = long.drop(long.index[(long["symbol"] == "BBB") & (long["date"] == DATES[7])])
    panel = build_panel(long)
    pd.testing.assert_frame_equal(panel.universe, panel.close.notna())
    assert not panel.universe.loc[DATES[7], "BBB"]


def test_a_supplied_universe_is_reindexed_and_masked_by_price() -> None:
    long = _long_table()
    long = long.drop(long.index[(long["symbol"] == "BBB") & (long["date"] == DATES[7])])
    wide = pd.DataFrame(True, index=DATES, columns=SYMBOLS + ["ZZZ"])

    panel = build_panel(long, universe=wide)
    # Reindexed onto the panel: unknown symbols dropped, unknown dates excluded.
    assert panel.universe.columns.tolist() == SYMBOLS
    assert panel.universe.dtypes.eq(bool).all()
    # A name with no price is not investable, whatever the universe claims.
    assert not panel.universe.loc[DATES[7], "BBB"]
    assert panel.universe.loc[DATES[8], "BBB"]


def test_industry_labels_are_tiled_across_every_date() -> None:
    panel = build_panel(_long_table())
    assert panel.industry.shape == (len(DATES), len(SYMBOLS))
    for sym, label in INDUSTRY.items():
        assert panel.industry[sym].unique().tolist() == [label]


@pytest.mark.parametrize("how", ["missing_column", "missing_label"])
def test_unlabelled_names_land_in_an_explicit_unknown_bucket(how: str) -> None:
    long = _long_table()
    if how == "missing_column":
        long = long.drop(columns=["industry"])
    else:
        long.loc[long["symbol"] == "BBB", "industry"] = np.nan

    panel = build_panel(long)
    if how == "missing_column":
        assert panel.industry.eq("Unknown").all().all()
    else:
        assert panel.industry["AAA"].iloc[0] == "Tech"
        assert panel.industry["BBB"].iloc[0] == "Unknown"


def test_duplicate_observations_keep_the_last_one() -> None:
    long = _long_table()
    extra = long.iloc[0].copy()
    extra["adj_close"] = 999.0
    panel = build_panel(pd.concat([long, extra.to_frame().T], ignore_index=True))
    assert panel.close.iloc[0]["AAA"] == pytest.approx(999.0)


def test_metadata_records_provenance() -> None:
    long = _long_table()
    assert build_panel(long).metadata["provenance"] == "TEST_FIXTURE"
    long.attrs = {}
    assert build_panel(long).metadata["provenance"] == "UNKNOWN"


def test_benchmark_is_carried_through() -> None:
    bench = pd.Series(np.linspace(100.0, 110.0, len(DATES)), index=DATES)
    panel = build_panel(_long_table(), benchmark=bench)
    pd.testing.assert_series_equal(panel.benchmark, bench)


# ----------------------------------------------------------------------
# MarketPanel helpers
# ----------------------------------------------------------------------
def test_tradable_masks_prices_to_the_universe() -> None:
    long = _long_table()
    long = long.drop(long.index[(long["symbol"] == "BBB") & (long["date"] == DATES[7])])
    panel = build_panel(long)
    pd.testing.assert_frame_equal(panel.tradable(), panel.close.where(panel.universe))
    assert pd.isna(panel.tradable().loc[DATES[7], "BBB"])


def test_forward_returns_are_aligned_to_the_signal_date() -> None:
    panel = build_panel(_long_table())
    fwd = panel.forward_returns(HORIZON)
    expected = panel.close.shift(-HORIZON) / panel.close - 1.0
    pd.testing.assert_frame_equal(fwd, expected)
    # The last `horizon` rows have no realised label yet.
    assert fwd.iloc[-1].isna().all()
    assert fwd.iloc[-(HORIZON + 1)].notna().all()


def test_forward_returns_honour_an_execution_lag() -> None:
    panel = build_panel(_long_table())
    lagged = panel.forward_returns(HORIZON, execution_lag=1)
    immediate = panel.forward_returns(HORIZON)
    # The window opens on the next session, so the day-one move is not in the
    # label - this is the difference that matters for a t+1 fill model.
    entry = panel.close.shift(-1)
    pd.testing.assert_frame_equal(lagged, panel.close.shift(-(HORIZON + 1)) / entry - 1.0)
    assert not lagged.iloc[0].equals(immediate.iloc[0])


def test_describe_summarises_the_panel() -> None:
    long = _long_table()
    long = long.drop(long.index[(long["symbol"] == "BBB") & (long["date"] == DATES[7])])
    summary = build_panel(long).describe()
    assert summary["dates"] == len(DATES)
    assert summary["symbols"] == len(SYMBOLS)
    assert summary["start"] == str(DATES.min().date())
    assert summary["end"] == str(DATES.max().date())
    assert summary["avg_breadth"] == pytest.approx((len(SYMBOLS) * len(DATES) - 1) / len(DATES))
    assert 0.0 < summary["pct_obs"] < 1.0


# ----------------------------------------------------------------------
# industry_dummies
# ----------------------------------------------------------------------
def test_dummies_drop_the_alphabetically_first_label() -> None:
    panel = build_panel(_long_table())
    dummies = industry_dummies(panel.industry)
    assert list(dummies) == ["Tech"]  # Energy is the omitted category
    assert dummies["Tech"].loc[DATES[0], "AAA"] == 1.0
    assert dummies["Tech"].loc[DATES[0], "BBB"] == 0.0


def test_dummies_keep_every_label_when_asked() -> None:
    panel = build_panel(_long_table())
    dummies = industry_dummies(panel.industry, drop_first=False)
    assert list(dummies) == ["Energy", "Tech"]
    assert dummies["Energy"].add(dummies["Tech"]).eq(1.0).all().all()


def test_dummies_ignore_non_string_labels() -> None:
    industry = pd.DataFrame([["Tech", np.nan]], index=DATES[:1], columns=["AAA", "BBB"])
    dummies = industry_dummies(industry, drop_first=False)
    # The unlabelled name gets no dummy of its own - it sits in the omitted
    # category, which is exactly why build_panel labels it "Unknown" instead.
    assert list(dummies) == ["Tech"]
    assert dummies["Tech"].iloc[0]["BBB"] == 0.0


def test_dummies_on_a_frame_without_labels() -> None:
    assert industry_dummies(pd.DataFrame()) == {}

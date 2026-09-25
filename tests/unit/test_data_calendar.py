"""Offline tests for the trading calendar and rebalance schedule.

Date alignment is the quietest way to break a backtest: an off-by-one in the
execution schedule does not raise, it just trades on the close that produced the
signal. ``execution_dates`` is described as "the structural guard against
look-ahead execution", so the tests assert the direction of the shift directly -
every execution date must be strictly *after* its signal date, and by exactly
``lag`` sessions.

One diagnostic was vacuous and is pinned here. ``execution_dates`` used to count
its in-sample signals as ``d <= calendar.max()`` over the dates ``next()``
returned, but ``next()`` clamps at the final session - so every execution date is
trivially within the sample and the log could only ever report N/N. With a lag of
10 on a 20-session calendar the truth is 3 of 5, and the log said 5 of 5. It now
counts the signals whose *unclamped* target session exists. The mapping itself is
unchanged: clamping onto the final session is a deliberate choice, not a bug.

Everything is deterministic: no network, no data provider, no shared state.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.data import calendar as calendar_module
from alphaforge.data.calendar import (
    TradingCalendar,
    embargo_mask,
    execution_dates,
    rebalance_dates,
)

CAL = pd.bdate_range("2024-01-01", periods=20)


# ----------------------------------------------------------------------
# TradingCalendar
# ----------------------------------------------------------------------
def test_the_calendar_sorts_and_deduplicates_its_input() -> None:
    shuffled = list(CAL[::-1]) + [CAL[3], CAL[3]]
    cal = TradingCalendar(shuffled)
    assert len(cal) == len(CAL)
    assert cal.index.is_monotonic_increasing
    assert cal.index.is_unique


def test_an_empty_calendar_is_rejected() -> None:
    """Every other method assumes at least one session exists."""
    with pytest.raises(ValueError, match="at least one date"):
        TradingCalendar([])


def test_len_and_index_agree() -> None:
    cal = TradingCalendar(CAL)
    assert len(cal) == len(CAL)
    assert list(cal.index) == list(CAL)


def test_position_of_a_session_is_its_offset() -> None:
    cal = TradingCalendar(CAL)
    for i in (0, 5, len(CAL) - 1):
        assert cal.position(CAL[i]) == i


def test_position_of_a_non_session_is_the_next_one() -> None:
    """2024-01-06 is a Saturday; the next session is Monday the 8th."""
    cal = TradingCalendar(CAL)
    assert cal.position(pd.Timestamp("2024-01-06")) == 5
    assert cal.index[5] == pd.Timestamp("2024-01-08")


def test_position_accepts_a_string() -> None:
    assert TradingCalendar(CAL).position("2024-01-01") == 0


def test_next_and_previous_walk_the_expected_number_of_sessions() -> None:
    cal = TradingCalendar(CAL)
    assert cal.next(CAL[0], 1) == CAL[1]
    assert cal.next(CAL[0], 4) == CAL[4]
    assert cal.previous(CAL[10], 3) == CAL[7]
    assert cal.next(CAL[3], 0) == CAL[3]


def test_next_and_previous_clamp_at_the_edges() -> None:
    cal = TradingCalendar(CAL)
    assert cal.next(CAL[-1], 5) == CAL[-1]
    assert cal.previous(CAL[0], 5) == CAL[0]


def test_previous_of_a_non_session_is_the_last_session_before_it() -> None:
    cal = TradingCalendar(CAL)
    assert cal.previous(pd.Timestamp("2024-01-06"), 1) == pd.Timestamp("2024-01-05")


def test_slice_bounds_are_inclusive() -> None:
    cal = TradingCalendar(CAL)
    got = cal.slice(CAL[3], CAL[7])
    assert list(got) == list(CAL[3:8])


def test_slice_is_open_ended() -> None:
    cal = TradingCalendar(CAL)
    assert list(cal.slice(start=CAL[15])) == list(CAL[15:])
    assert list(cal.slice(end=CAL[3])) == list(CAL[:4])
    assert len(cal.slice()) == len(CAL)


def test_slice_outside_the_calendar_is_empty() -> None:
    cal = TradingCalendar(CAL)
    assert len(cal.slice(pd.Timestamp("2030-01-01"), pd.Timestamp("2030-12-31"))) == 0


# ----------------------------------------------------------------------
# rebalance_dates
# ----------------------------------------------------------------------
def test_an_empty_calendar_yields_an_empty_schedule() -> None:
    assert len(rebalance_dates(pd.DatetimeIndex([]))) == 0


def test_daily_rebalancing_returns_the_calendar_itself() -> None:
    assert list(rebalance_dates(CAL, "daily")) == list(CAL)


def test_the_schedule_is_the_last_session_of_each_period() -> None:
    """20 business days from Mon 2024-01-01 span four weeks, not three."""
    got = rebalance_dates(CAL, "weekly")
    assert list(got) == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-12"),
        pd.Timestamp("2024-01-19"),
        pd.Timestamp("2024-01-26"),
    ]


def test_the_schedule_only_contains_real_sessions() -> None:
    """A period end that is not a trading day must snap back to one."""
    year = pd.bdate_range("2024-01-01", periods=260)
    for freq in ("weekly", "monthly", "quarterly", "yearly"):
        got = rebalance_dates(year, freq)
        assert set(got) <= set(year), f"{freq} produced a non-session"


def test_monthly_rebalancing_gives_one_date_per_month() -> None:
    year = pd.bdate_range("2024-01-01", periods=260)
    got = rebalance_dates(year, freq="monthly")
    assert len(got) == 12
    assert got.is_monotonic_increasing
    assert len(set(got.to_period("M"))) == 12


def test_the_frequency_is_case_insensitive() -> None:
    year = pd.bdate_range("2024-01-01", periods=260)
    assert list(rebalance_dates(year, "MONTHLY")) == list(rebalance_dates(year, "monthly"))


def test_an_unknown_frequency_falls_back_to_monthly() -> None:
    """Documented default; silently rebalancing daily would be far worse."""
    year = pd.bdate_range("2024-01-01", periods=260)
    assert list(rebalance_dates(year, "fortnightly")) == list(rebalance_dates(year, "monthly"))


# ----------------------------------------------------------------------
# execution_dates - the look-ahead guard
# ----------------------------------------------------------------------
def test_execution_is_strictly_after_the_signal() -> None:
    signals = CAL[::4]
    got = execution_dates(signals, CAL, lag=1)
    for signal, execution in got.items():
        assert execution > signal, "a signal must never trade on its own close"
        assert execution == CAL[CAL.get_loc(signal) + 1]


def test_the_lag_is_honoured_exactly() -> None:
    signals = CAL[::5]
    for lag in (1, 2, 3):
        got = execution_dates(signals, CAL, lag=lag)
        for signal, execution in got.items():
            expected = CAL[min(CAL.get_loc(signal) + lag, len(CAL) - 1)]
            assert execution == expected


def test_the_result_is_indexed_by_signal_date() -> None:
    signals = CAL[::4]
    got = execution_dates(signals, CAL)
    assert list(got.index) == list(signals)
    assert got.name == "execution_date"


def test_a_signal_too_close_to_the_end_is_clamped_not_dropped() -> None:
    """The mapping is deliberately clamped; the count below is what changed."""
    got = execution_dates(CAL[-1:], CAL, lag=5)
    assert got.iloc[0] == CAL[-1]


def _capture_warnings(monkeypatch) -> list[str]:
    """The project logger does not propagate to the root logger, so patch it."""
    seen: list[str] = []
    monkeypatch.setattr(calendar_module.log, "warning", lambda msg: seen.append(str(msg)))
    monkeypatch.setattr(calendar_module.log, "debug", lambda msg: None)
    return seen


def test_the_in_sample_count_is_not_vacuous(monkeypatch) -> None:
    """Regression: the count used to be ``d <= calendar.max()`` on clamped dates.

    ``next`` clamps at the final session, so that comparison could only ever
    report N/N - with a lag of 10 on a 20-session calendar the truth is 3 of 5
    and the log claimed 5 of 5.
    """
    seen = _capture_warnings(monkeypatch)
    execution_dates(CAL[::4], CAL, lag=10)
    assert any("3/5" in m for m in seen), f"expected 3 of 5 in-sample signals, got: {seen}"


def test_a_fully_executable_schedule_is_not_warned_about(monkeypatch) -> None:
    seen = _capture_warnings(monkeypatch)
    execution_dates(CAL[::4], CAL, lag=1)
    assert seen == []


def test_execution_dates_rejects_an_empty_calendar() -> None:
    with pytest.raises(ValueError, match="at least one date"):
        execution_dates(pd.DatetimeIndex([]), pd.DatetimeIndex([]))


# ----------------------------------------------------------------------
# embargo_mask
# ----------------------------------------------------------------------
def test_the_embargo_blocks_a_window_after_the_training_end() -> None:
    got = embargo_mask(CAL[9], CAL, embargo_days=5)
    blocked = CAL[~got.to_numpy()]
    assert list(blocked) == list(CAL[9:14])
    assert got.sum() == len(CAL) - 5


def test_the_embargo_includes_the_training_end_itself() -> None:
    """Conservative: train_end is a training date, so it can never be tested on."""
    got = embargo_mask(CAL[9], CAL, embargo_days=3)
    assert not got.loc[CAL[9]]
    assert got.loc[CAL[8]]
    assert got.loc[CAL[12]]


def test_a_zero_day_embargo_blocks_nothing() -> None:
    assert embargo_mask(CAL[9], CAL, embargo_days=0).all()


def test_an_embargo_past_the_end_is_clamped() -> None:
    got = embargo_mask(CAL[15], CAL, embargo_days=1000)
    assert not got.loc[CAL[15] :].any()
    assert got.loc[: CAL[14]].all()


def test_a_training_end_before_the_calendar_blocks_everything() -> None:
    """No overlap means no session is safely outside the window."""
    assert not embargo_mask(pd.Timestamp("2000-01-01"), CAL, embargo_days=5).any()


def test_the_mask_is_indexed_like_the_calendar() -> None:
    got = embargo_mask(CAL[9], CAL, embargo_days=2)
    assert list(got.index) == list(CAL)
    assert got.dtype == bool

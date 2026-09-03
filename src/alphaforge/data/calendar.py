"""Trading calendar and rebalance-schedule generation.

Rebalance dates are always derived from the *observed* calendar, and execution
is deliberately scheduled on the session **after** the signal date
(see :func:`execution_dates`) so a signal can never be traded on the close that
produced it.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("data.calendar")


class TradingCalendar:
    """A calendar derived from actual observations (no hardcoded holiday lists)."""

    def __init__(self, dates: Iterable[pd.Timestamp]) -> None:
        self._index = pd.DatetimeIndex(pd.to_datetime(sorted(pd.unique(pd.Index(list(dates))))))
        if len(self._index) == 0:
            raise ValueError("TradingCalendar requires at least one date")
        self._pos = pd.Series(np.arange(len(self._index)), index=self._index)

    def __len__(self) -> int:
        return len(self._index)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    def position(self, ts) -> int:
        ts = pd.Timestamp(ts)
        if ts in self._pos.index:
            return int(self._pos.loc[ts])
        return int(self._pos.index.searchsorted(ts))

    def next(self, ts, n: int = 1) -> pd.Timestamp:
        """The session ``n`` steps after ``ts`` (clamped at the end)."""
        pos = min(self.position(ts) + n, len(self._index) - 1)
        return self._index[pos]

    def previous(self, ts, n: int = 1) -> pd.Timestamp:
        pos = max(self.position(ts) - n, 0)
        return self._index[pos]

    def slice(self, start=None, end=None) -> pd.DatetimeIndex:
        idx = self._index
        if start is not None:
            idx = idx[idx >= pd.Timestamp(start)]
        if end is not None:
            idx = idx[idx <= pd.Timestamp(end)]
        return idx


def rebalance_dates(calendar: pd.DatetimeIndex, freq: str = "monthly") -> pd.DatetimeIndex:
    """Last trading session of each period implied by ``freq``."""
    if len(calendar) == 0:
        return pd.DatetimeIndex([])
    s = pd.Series(calendar, index=calendar)
    rule = {"daily": "D", "weekly": "W", "monthly": "ME", "quarterly": "QE", "yearly": "YE"}.get(
        freq.lower(), "ME"
    )
    if freq.lower() == "daily":
        return pd.DatetimeIndex(calendar)
    grouped = s.groupby(pd.Grouper(freq=rule)).max()
    out = grouped.dropna().astype("datetime64[ns]")
    return pd.DatetimeIndex(sorted(set(out) & set(calendar)))


def execution_dates(signal_dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex, lag: int = 1):
    """Map signal dates to execution dates, shifted forward by ``lag`` sessions.

    This single function is the structural guard against look-ahead execution:
    every backtest must trade on the returned dates, never on ``signal_dates``.
    """
    cal = TradingCalendar(calendar)
    exec_dates = [cal.next(d, lag) for d in signal_dates]
    lag_ok = [d <= calendar.max() for d in exec_dates]
    out = pd.Series(exec_dates, index=signal_dates, name="execution_date")
    log.debug(f"execution_dates: lag={lag} sessions, {sum(lag_ok)}/{len(lag_ok)} within sample")
    return out


def embargo_mask(
    train_end: pd.Timestamp,
    all_dates: pd.DatetimeIndex,
    embargo_days: int,
) -> pd.Series:
    """Boolean mask of dates that are safely outside the embargo window."""
    train_end = pd.Timestamp(train_end)
    cutoff = all_dates[all_dates <= train_end]
    if len(cutoff) == 0:
        return pd.Series(False, index=all_dates)
    start = cutoff[-1]
    pos = all_dates.searchsorted(start)
    end_pos = min(pos + embargo_days, len(all_dates))
    blocked = set(all_dates[pos:end_pos])
    return pd.Series([d not in blocked for d in all_dates], index=all_dates)


__all__ = ["TradingCalendar", "rebalance_dates", "execution_dates", "embargo_mask"]

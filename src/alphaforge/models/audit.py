"""Export and replay walk-forward boundaries on the actual label calendar.

The splitter operates on dates retained in the supervised dataset. Labels use
row shifts on the original market panel. Retain both calendars so a missing
dataset date cannot silently change the meaning of an H-session label.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from alphaforge.models.split import WalkForwardConfig, WalkForwardSplitter


def _audit(calendar: pd.DatetimeIndex, counts: list[int], horizon: int, config: dict) -> dict:
    if (
        calendar.empty
        or calendar.has_duplicates
        or calendar.hasnans
        or not calendar.is_monotonic_increasing
        or calendar.tz is not None
        or not calendar.equals(calendar.normalize())
    ):
        raise ValueError("Split audit requires unique, chronological, timezone-naive session dates")
    if len(counts) != len(calendar) or any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Split audit sample counts must be nonnegative integers on the calendar")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Split audit label horizon must be a positive integer")
    if set(config) != set(asdict(WalkForwardConfig())):
        raise ValueError("Split audit walk-forward fields differ")
    for name in ("train_years", "test_years", "step_years", "min_train_days"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"Split audit {name} must be a positive integer")
    for name in ("purge_days", "embargo_days"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"Split audit {name} must be a nonnegative integer")
    if type(config["expanding"]) is not bool:
        raise ValueError("Split audit expanding must be boolean")
    positions = np.flatnonzero(np.asarray(counts) > 0)
    if positions.size == 0 or positions[-1] + horizon >= len(calendar):
        raise ValueError("Split audit dataset rows must have a complete forward-label calendar")
    dates = calendar[positions]
    sample_counts = np.asarray(counts, dtype=np.int64)[positions]
    folds = WalkForwardSplitter(WalkForwardConfig(**config)).split(dates)
    if not folds:
        raise ValueError("Split audit has no usable walk-forward folds")
    records = []
    seen: set[int] = set()
    for fold in folds:
        train = positions[fold.train_mask]
        test = positions[fold.test_mask]
        label_end = int(train[-1]) + horizon
        if label_end >= int(test[0]):
            raise ValueError(
                f"Training label overlaps test in fold {fold.index}: "
                f"{calendar[label_end].date()} >= {calendar[test[0]].date()}"
            )
        train_samples = int(sample_counts[fold.train_mask].sum())
        test_samples = int(sample_counts[fold.test_mask].sum())
        if train_samples < 100 or test_samples < 10:
            raise ValueError("Split audit refuses folds the model would silently skip")
        if seen.intersection(test.tolist()):
            raise ValueError("Split audit test sessions overlap between folds")
        seen.update(test.tolist())
        records.append(
            {
                "fold": fold.index,
                "train_start": str(calendar[train[0]].date()),
                "train_end": str(calendar[train[-1]].date()),
                "last_train_label_end": str(calendar[label_end].date()),
                "test_start": str(calendar[test[0]].date()),
                "test_end": str(calendar[test[-1]].date()),
                "train_sessions": len(train),
                "test_sessions": len(test),
                "train_samples": train_samples,
                "test_samples": test_samples,
                "gap_sessions": int(test[0] - train[-1] - 1),
            }
        )
    return {
        "schema_version": 1,
        "label_horizon_sessions": horizon,
        "market_sessions": [str(day.date()) for day in calendar],
        "samples_per_session": counts,
        "walk_forward": config,
        "folds": records,
    }


def build_split_audit(dataset, market_dates, config: WalkForwardConfig) -> dict:
    """Check real train-label endpoints before any comparison models are fitted."""
    calendar = pd.DatetimeIndex(market_dates)
    row_dates = pd.DatetimeIndex(dataset.dates)
    if row_dates.hasnans or not row_dates.isin(calendar).all():
        raise ValueError("Split audit dataset dates leave the market calendar")
    counts = pd.Series(1, index=row_dates).groupby(level=0).sum().reindex(calendar, fill_value=0)
    return _audit(
        calendar,
        [int(value) for value in counts],
        dataset.metadata["horizon"],
        asdict(config),
    )


def verify_split_audit(record: dict) -> dict:
    """Reconstruct masks/counts and recheck label boundaries from saved calendars.

    This verifies internal consistency. A rewritten calendar requires an
    independently trusted source identity to detect; this is not a signature.
    """
    expected_fields = {
        "schema_version",
        "label_horizon_sessions",
        "market_sessions",
        "samples_per_session",
        "walk_forward",
        "folds",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected_fields
        or record["schema_version"] != 1
    ):
        raise ValueError("Unsupported split audit schema")
    if not isinstance(record["market_sessions"], list) or any(
        not isinstance(day, str) for day in record["market_sessions"]
    ):
        raise ValueError("Split audit sessions must be ISO date strings")
    try:
        calendar = pd.DatetimeIndex(
            pd.to_datetime(record["market_sessions"], format="%Y-%m-%d", errors="raise")
        )
        expected = _audit(
            calendar,
            record["samples_per_session"],
            record["label_horizon_sessions"],
            record["walk_forward"],
        )
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Invalid split audit calendar or configuration") from error
    if record != expected:
        raise ValueError(
            "Split audit differs from reconstructed session masks and label boundaries"
        )
    return {
        "folds": len(expected["folds"]),
        "samples": sum(expected["samples_per_session"]),
        "oos_forecasts": sum(fold["test_samples"] for fold in expected["folds"]),
        "forecast_start": expected["folds"][0]["test_start"],
        "forecast_end": expected["folds"][-1]["test_end"],
    }

"""The saved date proof must test label endpoints, rather than repeat a setting."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphaforge.models.audit import build_split_audit, verify_split_audit
from alphaforge.models.split import WalkForwardConfig


def fixture(horizon=21, missing=None, names=2):
    calendar = pd.bdate_range("2018-01-01", "2021-03-31")
    observed = calendar[:-horizon]
    if missing is not None:
        observed = observed.difference(pd.DatetimeIndex(missing))
    dataset = SimpleNamespace(
        dates=pd.Series(np.repeat(observed, names)), metadata={"horizon": horizon}
    )
    config = WalkForwardConfig(
        train_years=1, purge_days=horizon, embargo_days=horizon, min_train_days=200
    )
    return calendar, dataset, config


def test_label_endpoints_and_sample_denominators_use_the_market_calendar():
    calendar, dataset, config = fixture()
    record = build_split_audit(dataset, calendar, config)
    result = verify_split_audit(record)
    assert result["samples"] == len(dataset.dates)
    assert result["folds"] == 3
    for fold in record["folds"]:
        position = calendar.get_loc(pd.Timestamp(fold["train_end"]))
        assert fold["last_train_label_end"] == str(calendar[position + 21].date())
        assert fold["last_train_label_end"] < fold["test_start"]
        assert fold["gap_sessions"] >= 21
        assert fold["test_samples"] == 2 * fold["test_sessions"]


def test_missing_supervised_dates_do_not_redefine_forward_label_sessions():
    missing = pd.bdate_range("2018-12-03", "2018-12-07")
    calendar, dataset, config = fixture(missing=missing)
    record = build_split_audit(dataset, calendar, config)
    first = record["folds"][0]
    expected = calendar[calendar.get_loc(pd.Timestamp(first["train_end"])) + 21]
    assert first["last_train_label_end"] == str(expected.date())
    assert verify_split_audit(record)["folds"] == 3


@pytest.mark.parametrize("purge", [0, 14, 20])
def test_underpurged_training_labels_fail_before_models_run(purge):
    calendar, dataset, config = fixture()
    config.purge_days, config.embargo_days = purge, 0
    with pytest.raises(ValueError, match="Training label overlaps test"):
        build_split_audit(dataset, calendar, config)


@pytest.mark.parametrize(
    "field,value",
    [("gap_sessions", 999), ("train_samples", 1), ("last_train_label_end", "2010-01-01")],
)
def test_rewritten_fold_evidence_fails_independent_boundary_reconstruction(field, value):
    calendar, dataset, config = fixture()
    record = build_split_audit(dataset, calendar, config)
    record["folds"][0][field] = value
    with pytest.raises(ValueError, match="differs from reconstructed"):
        verify_split_audit(record)


def test_trailing_rows_without_observable_labels_cannot_enter_the_audit():
    calendar, dataset, config = fixture()
    dataset.dates = pd.concat([dataset.dates, pd.Series(calendar[-1:])], ignore_index=True)
    with pytest.raises(ValueError, match="complete forward-label calendar"):
        build_split_audit(dataset, calendar, config)


def test_unknown_market_dates_are_rejected():
    calendar, dataset, config = fixture()
    dataset.dates.iloc[0] = pd.Timestamp("2017-01-01")
    with pytest.raises(ValueError, match="leave the market calendar"):
        build_split_audit(dataset, calendar, config)


def test_small_folds_cannot_be_silently_dropped_from_a_comparison():
    calendar, dataset, config = fixture(names=1)
    calendar = calendar[calendar <= "2019-02-05"]
    dataset.dates = dataset.dates[dataset.dates.isin(calendar[:-21])]
    with pytest.raises(ValueError, match="silently skip"):
        build_split_audit(dataset, calendar, config)


def test_overlapping_test_years_are_rejected_before_fitting():
    calendar, dataset, config = fixture()
    config.test_years = 2
    with pytest.raises(ValueError, match="test sessions overlap"):
        build_split_audit(dataset, calendar, config)


def test_boolean_counts_are_not_accepted_as_a_one_row_session():
    calendar, dataset, config = fixture()
    record = copy.deepcopy(build_split_audit(dataset, calendar, config))
    record["samples_per_session"][0] = True
    with pytest.raises(ValueError, match="sample counts"):
        verify_split_audit(record)

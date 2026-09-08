"""Offline tests for the time-series cross-validation splitters.

A leaky splitter is the most expensive defect in the whole repo: it does not
crash, it does not warn, it just produces a backtest that cannot lose. These
tests therefore lock the *leakage contract* rather than the shape of the return
value:

  * training data always ends before the test block opens;
  * with a ``H``-day label, no training observation survives whose label is still
    forming at ``t + H`` when the test block opens (purge);
  * no test observation sits within ``embargo_days`` of a training observation,
    on **either** edge of the test block (embargo);
  * a fold whose training window starves after purging is dropped, never emitted
    half-empty.

Everything is deterministic: no network, no randomness, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.models.split import (
    PurgedKFold,
    WalkForwardConfig,
    WalkForwardSplitter,
)

# Eleven years of business days: long enough for a four-year expanding window to
# produce several one-year test folds even after purging and embargoing.
DATES = pd.bdate_range("2014-01-02", "2024-12-31")
HORIZON = 21


def _split(dates: pd.DatetimeIndex = DATES, **overrides: object) -> list:
    """Split ``dates`` with ``overrides`` layered on the default config."""
    return WalkForwardSplitter(WalkForwardConfig(**overrides)).split(dates)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# WalkForwardSplitter
# ----------------------------------------------------------------------
def test_no_dates_returns_no_folds() -> None:
    assert WalkForwardSplitter().split(pd.DatetimeIndex([])) == []


def test_dates_are_sorted_and_deduplicated_before_splitting() -> None:
    scrambled = pd.DatetimeIndex(list(DATES[::-1]) + list(DATES[:10]))
    folds = WalkForwardSplitter().split(scrambled)
    baseline = WalkForwardSplitter().split(DATES)
    assert [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds] == [
        (f.train_start, f.train_end, f.test_start, f.test_end) for f in baseline
    ]


def test_training_always_ends_before_testing_begins() -> None:
    folds = _split()
    assert folds
    for fold in folds:
        assert fold.train_end < fold.test_start


def test_folds_advance_through_time_and_are_numbered_in_order() -> None:
    folds = _split()
    for previous, nxt in zip(folds, folds[1:]):
        assert nxt.index == previous.index + 1
        assert nxt.test_start > previous.test_start
        assert nxt.test_end > previous.test_end


def test_fold_boundaries_agree_with_their_masks() -> None:
    for fold in _split():
        assert DATES[fold.train_mask][0] == fold.train_start
        assert DATES[fold.train_mask][-1] == fold.train_end
        assert DATES[fold.test_mask][0] == fold.test_start
        assert DATES[fold.test_mask][-1] == fold.test_end
        # The training block is one contiguous run of sessions.
        first = int(DATES.get_loc(fold.train_start))
        last = int(DATES.get_loc(fold.train_end))
        assert int(fold.train_mask.sum()) == last - first + 1


def test_masks_are_disjoint_and_both_non_empty() -> None:
    for fold in _split():
        assert not np.any(fold.train_mask & fold.test_mask)
        assert fold.train_mask.sum() > 0
        assert fold.test_mask.sum() > 0


def test_without_purge_the_training_tail_touches_the_test_block() -> None:
    """Control for the purge test: the gap is *not* there by accident."""
    folds = _split(purge_days=0, embargo_days=0)
    assert folds
    for fold in folds:
        assert (fold.test_start - fold.train_end).days < 7


def test_purge_removes_the_training_labels_that_overlap_the_test_block() -> None:
    folds = _split(purge_days=HORIZON, embargo_days=0)
    assert folds
    for fold in folds:
        # The label of the last surviving training sample is realised at
        # ``train_end + HORIZON`` and must still fall strictly before the test
        # block opens - otherwise the fold trains on the future.
        assert fold.train_end + pd.Timedelta(days=HORIZON) < fold.test_start


def test_embargo_forces_a_gap_after_the_last_training_session() -> None:
    folds = _split(purge_days=0, embargo_days=HORIZON)
    assert folds
    for fold in folds:
        assert (fold.test_start - fold.train_end).days >= HORIZON


def test_expanding_window_keeps_the_earliest_observation() -> None:
    folds = _split(expanding=True)
    assert {fold.train_start for fold in folds} == {DATES.min()}


def test_rolling_window_drops_stale_history() -> None:
    folds = _split(expanding=False, train_years=2)
    assert len(folds) > 1
    assert folds[0].train_start < folds[-1].train_start
    for fold in folds:
        assert (fold.train_end - fold.train_start).days <= 2 * 366


def test_step_years_controls_how_often_a_fold_is_taken() -> None:
    assert len(_split(step_years=1)) == 7
    assert len(_split(step_years=2)) == 4


def test_test_years_widens_the_test_block() -> None:
    one_year = _split(test_years=1, purge_days=0, embargo_days=0)
    two_years = _split(test_years=2, purge_days=0, embargo_days=0)
    narrow = (one_year[0].test_end - one_year[0].test_start).days
    wide = (two_years[0].test_end - two_years[0].test_start).days
    assert wide > narrow * 1.8


def test_min_train_days_rejects_a_short_history() -> None:
    assert _split(min_train_days=10_000_000) == []


def test_a_fold_is_dropped_when_purging_starves_the_training_window() -> None:
    # A sample with exactly one fold: train on 2014-2017, test on 2018.
    dates = pd.bdate_range("2014-01-02", "2018-12-31")
    unpurged = _split(dates, purge_days=0, embargo_days=0)
    assert len(unpurged) == 1
    n_train = int(unpurged[0].train_mask.sum())

    # ``n_train`` passes the pre-purge check and fails the post-purge one, so the
    # fold must disappear rather than be emitted with a truncated window.
    starved = _split(dates, purge_days=HORIZON, embargo_days=0, min_train_days=n_train)
    assert starved == []


def test_degenerate_single_observation_panel_yields_nothing() -> None:
    # A zero-length training window would start exactly at the end of the
    # sample; the splitter must stop instead of emitting that fold.
    dates = pd.DatetimeIndex(["2020-01-01"])
    assert _split(dates, train_years=0, expanding=False) == []


def test_get_n_splits_is_unknown_until_the_dates_are_seen() -> None:
    assert WalkForwardSplitter().get_n_splits() == -1


def test_fold_repr_reports_its_span_and_sizes() -> None:
    fold = _split()[0]
    text = repr(fold)
    assert "Fold(0" in text
    assert f"n_train={int(fold.train_mask.sum())}" in text
    assert f"n_test={int(fold.test_mask.sum())}" in text


# ----------------------------------------------------------------------
# PurgedKFold
# ----------------------------------------------------------------------
def test_purged_kfold_emits_one_split_per_fold() -> None:
    # Regression: the purge/embargo helpers used to anchor on the *last training
    # date*, which for every fold but the last sits at the far right of the
    # sample. That emptied the test block of 4 folds out of 5, silently turning
    # cross-validation into a single split.
    assert len(PurgedKFold(n_splits=5).split(DATES)) == 5
    assert len(PurgedKFold(n_splits=3).split(DATES)) == 3


def test_purged_kfold_masks_are_disjoint_and_non_empty() -> None:
    splits = PurgedKFold(n_splits=4).split(DATES)
    assert splits
    for train, test in splits:
        assert not np.any(train & test)
        assert train.sum() > 0
        assert test.sum() > 0


def test_purged_kfold_purges_the_training_observations_next_to_the_test_block() -> None:
    for train, test in PurgedKFold(n_splits=4, purge_days=HORIZON, embargo_days=0).split(DATES):
        # A training session whose label is still forming when the test block
        # opens sits in [test_first - HORIZON, test_last]; none may survive.
        first, last = np.where(test)[0][0], np.where(test)[0][-1]
        start = int(DATES.searchsorted(DATES[first] - pd.Timedelta(days=HORIZON)))
        assert not train[start : last + 1].any()


def test_purged_kfold_embargoes_both_edges_of_the_test_block() -> None:
    for train, test in PurgedKFold(n_splits=4, purge_days=0, embargo_days=HORIZON).split(DATES):
        train_pos = np.where(train)[0]
        test_pos = np.where(test)[0]
        before = train_pos[train_pos < test_pos[0]]
        after = train_pos[train_pos > test_pos[-1]]
        # Whichever side of the test block the training set touches, the seam
        # must stay clean - a K-fold training set wraps around on both sides.
        if before.size:
            assert (DATES[test_pos[0]] - DATES[before[-1]]).days >= HORIZON
        if after.size:
            assert (DATES[after[0]] - DATES[test_pos[-1]]).days >= HORIZON


def test_purged_kfold_without_purge_or_embargo_partitions_the_sample() -> None:
    splits = PurgedKFold(n_splits=5, purge_days=0, embargo_days=0).split(DATES)
    assert len(splits) == 5
    covered = np.zeros(len(DATES), dtype=bool)
    for train, test in splits:
        assert not np.any(train & test)
        assert np.all(train | test)
        covered |= test
    # Every observation is tested exactly once across the folds.
    assert covered.all()


def test_purged_kfold_drops_folds_that_empty_out() -> None:
    # Three sessions cannot survive a 21-day purge and a 10-day embargo.
    dates = pd.bdate_range("2024-01-01", periods=3)
    assert PurgedKFold(n_splits=5, purge_days=HORIZON, embargo_days=10).split(dates) == []


def test_purged_kfold_with_no_training_data_at_all() -> None:
    # One fold leaves nothing to train on; the helpers must not blow up on the
    # empty mask and the fold must be dropped.
    assert PurgedKFold(n_splits=1).split(DATES) == []

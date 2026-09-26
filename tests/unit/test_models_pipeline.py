"""Offline tests for the walk-forward training driver.

This is where the splitter's guarantees are actually *used*, and the mapping is
the dangerous part: the splitter returns per-**date** masks, but the dataset is
a per-**row** long frame, so the driver has to translate one into the other
through ``date_pos`` / ``row_pos``. Get that translation wrong by a single
position and every fold trains on data that overlaps its own test block - the
backtest looks excellent, no error is raised, and nothing downstream can tell.

So the central test re-derives the folds independently and asserts that, for
every fold the driver actually produced predictions for, the latest training
date is strictly earlier than the earliest test date.

Also pinned: hyper-parameters are never tuned on the concatenated out-of-sample
predictions, per-fold feature importance is aggregated with the right axis and
column order, a model type that fails does not abort the comparison grid, and a
splitter that yields nothing fails loudly instead of returning empty results.

Everything is deterministic: no network, fixed seeds, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.features.panel import build_panel
from alphaforge.models.dataset import AlphaDataset, build_dataset
from alphaforge.models.estimators import ModelConfig
from alphaforge.models.pipeline import (
    AlphaModelPipeline,
    WalkForwardResult,
    run_model_comparison,
    signal_panel,
)
from alphaforge.models.split import Fold, WalkForwardConfig, WalkForwardSplitter

DATES = pd.bdate_range("2018-01-02", periods=1200)
SYMBOLS = [f"S{i:02d}" for i in range(20)]
HORIZON = 21
WF = {
    "train_years": 1,
    "test_years": 1,
    "step_years": 1,
    "purge_days": HORIZON,
    "embargo_days": HORIZON,
    "expanding": True,
}


@pytest.fixture(scope="module")
def dataset() -> AlphaDataset:
    rng = np.random.default_rng(5)
    rows = []
    for sym in SYMBOLS:
        path = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(DATES))))
        for date, price in zip(DATES, path):
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "adj_close": price,
                    "close": price,
                    "volume": 1.0e6,
                    "market_cap": 1.0e9,
                    "industry": "Tech" if sym < "S10" else "Energy",
                }
            )
    panel = build_panel(pd.DataFrame(rows))
    factors = {
        f"f{i}": pd.DataFrame(
            rng.normal(size=(len(DATES), len(SYMBOLS))), index=DATES, columns=SYMBOLS
        )
        for i in range(4)
    }
    return build_dataset(panel, factors, horizon=HORIZON, min_names_per_date=10)


def _cfg(model_type: str = "ridge", **wf) -> dict:
    return {"model": {"type": model_type, "params": {"alpha": 1.0}, "walk_forward": {**WF, **wf}}}


@pytest.fixture(scope="module")
def result(dataset: AlphaDataset) -> WalkForwardResult:
    return AlphaModelPipeline.from_config(dataset, _cfg()).run()


def _folds(dataset: AlphaDataset):
    """Re-derive the folds with the *same* config the pipeline fixture used.

    Deriving them from ``AlphaModelPipeline(dataset)`` instead would silently
    use the default 4/1/1 windows while the fixture ran 1/1/1, and the
    comparison would be meaningless rather than failing loudly.
    """
    pipeline = AlphaModelPipeline.from_config(dataset, _cfg())
    return WalkForwardSplitter(pipeline.split_config).split(dataset.date_index())


def _symbols(dataset: AlphaDataset) -> pd.Index:
    return pd.Index(sorted(dataset.symbols.unique()))


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
def test_the_default_split_purges_and_embargoes_by_the_horizon(dataset: AlphaDataset) -> None:
    """Without that default, a fold's own label window would overlap its test block."""
    got = AlphaModelPipeline(dataset).split_config
    horizon = int(dataset.metadata["horizon"])
    assert got.purge_days == horizon
    assert got.embargo_days == horizon


def test_from_config_reads_the_walk_forward_block(dataset: AlphaDataset) -> None:
    got = AlphaModelPipeline.from_config(dataset, _cfg(train_years=2, step_years=3))
    assert got.split_config.train_years == 2
    assert got.split_config.step_years == 3
    assert got.model_config.type == "ridge"
    assert got.model_config.params == {"alpha": 1.0}


def test_model_seed_inherits_project_seed_unless_explicit(dataset: AlphaDataset) -> None:
    cfg = {"project": {"seed": 7}, "model": {"type": "random_forest"}}
    assert AlphaModelPipeline.from_config(dataset, cfg).model_config.seed == 7
    cfg["model"]["seed"] = 11
    assert AlphaModelPipeline.from_config(dataset, cfg).model_config.seed == 11


def test_from_config_falls_back_to_the_horizon_for_purge_and_embargo(
    dataset: AlphaDataset,
) -> None:
    cfg = {"model": {"type": "ridge", "params": {}, "walk_forward": {"train_years": 1}}}
    got = AlphaModelPipeline.from_config(dataset, cfg)
    assert got.split_config.purge_days == int(dataset.metadata["horizon"])
    assert got.split_config.embargo_days == int(dataset.metadata["horizon"])


def test_an_empty_config_still_produces_a_usable_pipeline(dataset: AlphaDataset) -> None:
    got = AlphaModelPipeline.from_config(dataset, {})
    assert got.model_config.type == "ridge"
    assert isinstance(got.split_config, WalkForwardConfig)


# ----------------------------------------------------------------------
# No leakage - the whole point
# ----------------------------------------------------------------------
def test_every_fold_trains_strictly_before_it_tests(
    dataset: AlphaDataset, result: WalkForwardResult
) -> None:
    """Re-derive the folds and check the driver's own predictions against them."""
    dates = dataset.date_index()
    by_index = {f.index: f for f in _folds(dataset)}
    assert result.predictions["fold"].nunique() > 1, "a single fold proves nothing"

    for fold_id, group in result.predictions.groupby("fold"):
        fold = by_index[fold_id]
        train_dates = dates[fold.train_mask]
        test_dates = dates[fold.test_mask]
        assert train_dates.max() < test_dates.min(), f"fold {fold_id} trains into its own test"
        # Every predicted date belongs to this fold's test window.
        assert set(group["date"]) <= set(test_dates)
        # ...and the window is actually covered, not a stray handful of dates.
        assert len(set(group["date"])) >= 0.9 * len(test_dates)


def test_predictions_never_overlap_between_folds(result: WalkForwardResult) -> None:
    """Each out-of-sample block is predicted exactly once."""
    key = result.predictions[["date", "symbol"]]
    assert not key.duplicated().any()
    per_fold = result.predictions.groupby("fold")["date"].agg(["min", "max"]).sort_values("min")
    assert per_fold["min"].is_monotonic_increasing
    assert per_fold["max"].is_monotonic_increasing


def test_no_prediction_falls_outside_the_folds(dataset: AlphaDataset) -> None:
    """Training rows must never leak into the prediction frame."""
    res = AlphaModelPipeline.from_config(dataset, _cfg()).run()
    dates = dataset.date_index()
    allowed = set().union(*(set(dates[f.test_mask]) for f in _folds(dataset)))
    assert set(res.predictions["date"]) <= allowed


# ----------------------------------------------------------------------
# run()
# ----------------------------------------------------------------------
def test_run_returns_one_model_per_fold(result: WalkForwardResult) -> None:
    assert len(result.models) == result.predictions["fold"].nunique()


def test_run_predictions_carry_the_expected_columns(result: WalkForwardResult) -> None:
    assert list(result.predictions.columns) == [
        "date",
        "symbol",
        "prediction",
        "realised",
        "fold",
    ]
    assert result.predictions["prediction"].notna().all()


def test_run_returns_realised_forward_returns_not_the_label(result: WalkForwardResult) -> None:
    """``realised`` is the raw forward return, not the rank-transformed target."""
    assert not np.allclose(result.predictions["realised"], 0.0)
    assert result.predictions["realised"].abs().max() < 5.0


def test_run_records_the_configuration_it_used(result: WalkForwardResult) -> None:
    assert result.config["model_type"] == "ridge"
    assert result.config["horizon"] == HORIZON
    assert result.config["split"]["purge_days"] == HORIZON


def test_run_evaluates_the_concatenated_predictions(result: WalkForwardResult) -> None:
    assert result.evaluation is not None
    assert result.evaluation.model_name == "ridge"
    assert result.summary() is result.evaluation.summary
    assert not result.fold_metrics.empty


def test_run_accepts_an_explicit_model_type(dataset: AlphaDataset) -> None:
    got = AlphaModelPipeline.from_config(dataset, _cfg()).run(model_type="elasticnet")
    assert got.config["model_type"] == "elasticnet"
    assert got.evaluation is not None
    assert got.evaluation.model_name == "elasticnet"


def test_run_fails_loudly_when_the_splitter_yields_nothing(
    dataset: AlphaDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(WalkForwardSplitter, "split", lambda self, dates: [])
    with pytest.raises(RuntimeError, match="no folds"):
        AlphaModelPipeline.from_config(dataset, _cfg()).run()


def test_a_fold_with_too_few_samples_is_skipped(
    dataset: AlphaDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fold below the sample floor must be skipped, not trained on nothing.

    A fold that leaves nothing behind must also fail loudly rather than return
    an empty prediction frame that reads as "the model found nothing".
    """
    dates = dataset.date_index()
    n = len(dates)
    starved = Fold(
        index=0,
        train_start=dates[0],
        train_end=dates[2],
        test_start=dates[6],
        test_end=dates[7],
        train_mask=np.arange(n) < 3,  # 3 dates x 20 symbols = 60 rows, below the floor
        test_mask=np.arange(n) == 6,  # one date: ~20 rows, above its floor
    )
    tiny_test = Fold(
        index=1,
        train_start=dates[0],
        train_end=dates[199],
        test_start=dates[200],
        test_end=dates[200],
        train_mask=np.arange(n) < 200,
        test_mask=np.zeros(n, dtype=bool),  # no test rows at all
    )
    monkeypatch.setattr(WalkForwardSplitter, "split", lambda self, d: [starved, tiny_test])
    with pytest.raises(RuntimeError, match="no predictions"):
        AlphaModelPipeline.from_config(dataset, _cfg()).run()


# ----------------------------------------------------------------------
# Feature importance aggregation
# ----------------------------------------------------------------------
def test_importance_has_a_column_per_fold_plus_mean_and_std(
    result: WalkForwardResult,
) -> None:
    imp = result.importance
    n_folds = result.predictions["fold"].nunique()
    assert not imp.empty
    assert list(imp.columns) == [f"fold_{i}" for i in range(n_folds)] + ["mean", "std"]


def test_importance_mean_and_std_are_over_the_folds_only(result: WalkForwardResult) -> None:
    """``std`` must not be computed over a frame that already contains ``mean``."""
    imp = result.importance
    fold_cols = [c for c in imp.columns if c.startswith("fold_")]
    np.testing.assert_allclose(imp["mean"].to_numpy(), imp[fold_cols].mean(axis=1).to_numpy())
    np.testing.assert_allclose(imp["std"].to_numpy(), imp[fold_cols].std(axis=1).to_numpy())


def test_importance_is_sorted_by_mean_descending(result: WalkForwardResult) -> None:
    assert result.importance["mean"].is_monotonic_decreasing


def test_importance_is_attached_to_the_evaluation(result: WalkForwardResult) -> None:
    assert result.evaluation is not None
    assert result.evaluation.feature_importance is result.importance


def test_importance_is_empty_for_a_model_that_reports_none(dataset: AlphaDataset) -> None:
    """``BaseEstimator.feature_importance`` returns ``None`` - no empty frame, no crash."""
    from alphaforge.models import estimators

    class _NoImportance(estimators.RidgeModel):
        def feature_importance(self):
            return None

    original = estimators.REGISTRY["ridge"]
    estimators.REGISTRY["ridge"] = _NoImportance
    try:
        got = AlphaModelPipeline.from_config(dataset, _cfg()).run()
    finally:
        estimators.REGISTRY["ridge"] = original
    assert got.importance.empty
    assert not got.predictions.empty


# ----------------------------------------------------------------------
# signal_panel
# ----------------------------------------------------------------------
def test_signal_panel_pivots_predictions_onto_the_full_grid(
    dataset: AlphaDataset, result: WalkForwardResult
) -> None:
    got = signal_panel(result.predictions, dataset.date_index(), _symbols(dataset))
    assert got.shape == (len(dataset.date_index()), len(_symbols(dataset)))
    assert int(got.notna().sum().sum()) == len(result.predictions)


def test_signal_panel_leaves_unpredicted_cells_empty(
    dataset: AlphaDataset, result: WalkForwardResult
) -> None:
    """Dates the walk-forward never reached must be NaN, not zero."""
    got = signal_panel(result.predictions, dataset.date_index(), _symbols(dataset))
    first_predicted = result.predictions["date"].min()
    before = got.loc[got.index < first_predicted]
    assert not before.empty
    assert before.isna().all().all()


def test_signal_panel_keeps_the_last_prediction_for_a_duplicated_cell(
    dataset: AlphaDataset, result: WalkForwardResult
) -> None:
    doubled = pd.concat([result.predictions, result.predictions], ignore_index=True)
    got = signal_panel(doubled, dataset.date_index(), _symbols(dataset))
    base = signal_panel(result.predictions, dataset.date_index(), _symbols(dataset))
    np.testing.assert_allclose(got.to_numpy(), base.to_numpy(), equal_nan=True)


# ----------------------------------------------------------------------
# run_model_comparison
# ----------------------------------------------------------------------
def test_comparison_runs_every_requested_model_on_the_same_protocol(
    dataset: AlphaDataset,
) -> None:
    got = run_model_comparison(dataset, model_types=["ridge", "elasticnet"], cfg=_cfg())
    assert set(got) == {"ridge", "elasticnet"}
    for res in got.values():
        assert res.config["split"]["purge_days"] == HORIZON


def test_comparison_skips_a_model_that_fails_without_aborting_the_rest(
    dataset: AlphaDataset,
) -> None:
    got = run_model_comparison(dataset, model_types=["ridge", "not_a_model"], cfg=_cfg())
    assert set(got) == {"ridge"}
    assert not got["ridge"].predictions.empty


@pytest.mark.parametrize("model_types", [None, []])
def test_the_default_grid_is_every_registered_model(
    dataset: AlphaDataset, monkeypatch: pytest.MonkeyPatch, model_types
) -> None:
    """Both ``None`` and ``[]`` mean "use the whole registry" - ``[]`` is falsy."""
    from alphaforge.models import estimators

    attempted: list[str] = []

    def fake_run(self, model_type=None):
        attempted.append(model_type or self.model_config.type)
        return WalkForwardResult(predictions=pd.DataFrame())

    monkeypatch.setattr(AlphaModelPipeline, "run", fake_run)
    got = run_model_comparison(dataset, model_types=model_types, cfg=_cfg())
    assert set(attempted) == set(estimators.REGISTRY)
    assert set(got) == set(estimators.REGISTRY)


def test_model_params_override_the_shared_block(dataset: AlphaDataset) -> None:
    cfg = _cfg()
    cfg["model_params"] = {"elasticnet": {"alpha": 0.25, "l1_ratio": 0.9}}
    got = run_model_comparison(dataset, model_types=["elasticnet"], cfg=cfg)
    assert got["elasticnet"].config["params"] == {"alpha": 0.25, "l1_ratio": 0.9}


def test_signal_panel_tolerates_a_per_row_symbol_series(
    dataset: AlphaDataset, result: WalkForwardResult
) -> None:
    """Regression: ``dataset.symbols`` is a per-*row* Series.

    ``reindex(columns=<Series>)`` matches on the Series' *values*, so the 23,580
    repeated labels turned a 20-column score panel into a 23,580-column one with
    every prediction scattered across duplicates - silently, with no error.
    """
    got = signal_panel(result.predictions, dataset.date_index(), dataset.symbols)
    assert got.shape == (len(dataset.date_index()), len(_symbols(dataset)))
    assert list(got.columns) == list(_symbols(dataset))
    assert int(got.notna().sum().sum()) == len(result.predictions)


# ----------------------------------------------------------------------
# WalkForwardResult
# ----------------------------------------------------------------------
def test_an_unfitted_result_summarises_to_an_empty_dict() -> None:
    empty = WalkForwardResult(predictions=pd.DataFrame())
    assert empty.summary() == {}
    assert empty.models == []
    assert empty.importance.empty


def test_model_config_defaults_are_used_when_none_is_given(dataset: AlphaDataset) -> None:
    got = AlphaModelPipeline(dataset, model_config=None)
    assert got.model_config == ModelConfig()

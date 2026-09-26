"""Offline tests for out-of-sample model evaluation.

These metrics decide whether a model ships, so the failure mode that matters is
not a crash - it is a *plausible-looking* number. A RankIC that is silently
computed against the wrong axis, a quantile bucket that quietly collapses to one
bucket, a t-statistic that forgets the overlap correction: all of them look like
a working model right up until the P&L.

So the tests below pin arithmetic against independently computed values:

  * ``daily_rank_ic`` reproduces a brute-force per-date Spearman correlation;
  * quantile buckets are monotone when the prediction really does order the
    realised return, and a date too thin to fill every bucket is NaN rather than
    a bucket quietly absorbing its neighbours;
  * ``prediction_turnover`` returns a hand-computed value on a constructed case;
  * ``t_stat`` applies the ``n / horizon`` overlap correction and ``ann_long_short``
    annualises by ``periods_per_year / horizon``;
  * ``fold_metrics`` carries **exactly one** ``n_days`` column - it used to carry
    two of the same name, so ``folds["n_days"]`` returned a DataFrame.

Everything is deterministic: no network, fixed seeds, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.models.evaluation import (
    ModelEvaluation,
    compare_models,
    daily_rank_ic,
    evaluate_predictions,
    icir,
    prediction_turnover,
    quantile_returns,
    rank_ic,
    top_quantile_stats,
)

DATES = pd.bdate_range("2022-01-03", periods=500)
SYMBOLS = [f"S{i:02d}" for i in range(12)]


def _predictions(noise: float = 1.0, seed: int = 11) -> pd.DataFrame:
    """A long prediction frame whose signal is deliberately strong."""
    rng = np.random.default_rng(seed)
    frames = []
    for d in DATES:
        f = rng.normal(size=len(SYMBOLS))
        frames.append(
            pd.DataFrame(
                {
                    "date": d,
                    "symbol": SYMBOLS,
                    "prediction": f,
                    "realised": 0.6 * f + noise * rng.normal(size=len(SYMBOLS)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _brute_force_rank_ic(predictions: pd.DataFrame) -> pd.Series:
    """Per-date Spearman correlation, computed the slow obvious way."""
    out = {}
    for date, g in predictions.groupby("date"):
        out[date] = g["prediction"].rank().corr(g["realised"].rank())
    series = pd.Series(out, name="rank_ic")
    series.index.name = "date"
    return series


def _flat_predictions(
    rows: list[tuple[str, float, float]], date: str = "2024-01-02"
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Timestamp(date),
            "symbol": [r[0] for r in rows],
            "prediction": [r[1] for r in rows],
            "realised": [r[2] for r in rows],
        }
    )


# ----------------------------------------------------------------------
# rank_ic
# ----------------------------------------------------------------------
def test_rank_ic_is_one_for_a_perfect_ranking() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rank_ic(x, x * 10.0 + 3.0) == pytest.approx(1.0)


def test_rank_ic_is_minus_one_for_a_reversed_ranking() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rank_ic(x, -x) == pytest.approx(-1.0)


def test_rank_ic_matches_a_hand_computed_spearman() -> None:
    x = pd.Series([3.0, 1.0, 4.0, 1.5, 5.0])
    y = pd.Series([2.0, 2.5, 1.0, 4.0, 5.0])
    assert rank_ic(x, y) == pytest.approx(x.rank().corr(y.rank()))


def test_rank_ic_ignores_names_with_a_missing_side() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    assert rank_ic(x, y) == pytest.approx(1.0)


def test_rank_ic_is_nan_below_five_names() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(rank_ic(x, x))


def test_rank_ic_is_nan_when_a_side_has_no_variance() -> None:
    assert np.isnan(rank_ic(pd.Series([1.0] * 6), pd.Series([1.0, 2, 3, 4, 5, 6])))


# ----------------------------------------------------------------------
# daily_rank_ic
# ----------------------------------------------------------------------
def test_daily_rank_ic_reproduces_the_brute_force_computation() -> None:
    """The headline check: a wrong ``.where`` axis would silently corrupt this."""
    pred = _predictions()
    got = daily_rank_ic(pred)
    expected = _brute_force_rank_ic(pred)
    assert not got.isna().all(), "the vectorised path produced nothing"
    np.testing.assert_allclose(
        got.to_numpy(dtype=float), expected.to_numpy(dtype=float), atol=1e-12
    )
    assert got.name == "rank_ic"


def test_daily_rank_ic_is_a_strong_positive_number_for_a_strong_signal() -> None:
    assert daily_rank_ic(_predictions()).mean() > 0.3


def test_daily_rank_ic_skips_a_date_that_is_too_thin() -> None:
    pred = _predictions()
    thin = pred["date"].iloc[0]
    pred.loc[pred["date"] == thin, "realised"] = np.nan
    got = daily_rank_ic(pred)
    assert np.isnan(got.loc[thin])
    assert got.dropna().shape[0] == len(DATES) - 1


def test_daily_rank_ic_honours_min_names() -> None:
    pred = _predictions()
    assert daily_rank_ic(pred, min_names=len(SYMBOLS) + 1).isna().all()


def test_daily_rank_ic_keeps_the_last_row_for_a_duplicated_date_and_symbol() -> None:
    """``aggfunc="last"``: a restated prediction supersedes the earlier one."""
    pred = _predictions()
    baseline = daily_rank_ic(pred)
    first_date = pred["date"].iloc[0]

    wild = pred.iloc[[0]].copy()
    wild["prediction"] = 1e9

    # Prepended -> the original row is still the last one, so nothing moves.
    prepended = daily_rank_ic(pd.concat([wild, pred], ignore_index=True))
    np.testing.assert_allclose(
        prepended.to_numpy(dtype=float), baseline.to_numpy(dtype=float), atol=1e-12
    )

    # Appended -> the restatement wins and that one date's IC changes.
    appended = daily_rank_ic(pd.concat([pred, wild], ignore_index=True))
    assert appended.loc[first_date] != pytest.approx(baseline.loc[first_date])
    np.testing.assert_allclose(
        appended.drop(index=first_date).to_numpy(dtype=float),
        baseline.drop(index=first_date).to_numpy(dtype=float),
        atol=1e-12,
    )


# ----------------------------------------------------------------------
# icir
# ----------------------------------------------------------------------
def test_icir_is_mean_over_standard_deviation() -> None:
    ic = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert icir(ic) == pytest.approx(ic.mean() / ic.std())


def test_icir_drops_missing_periods_first() -> None:
    assert icir(pd.Series([0.1, 0.2, np.nan, 0.3, 0.4])) == pytest.approx(
        icir(pd.Series([0.1, 0.2, 0.3, 0.4]))
    )


def test_icir_is_nan_without_two_observations() -> None:
    assert np.isnan(icir(pd.Series([0.1])))
    assert np.isnan(icir(pd.Series([], dtype=float)))


def test_icir_is_nan_when_the_ic_never_moves() -> None:
    assert np.isnan(icir(pd.Series([0.1] * 5)))


# ----------------------------------------------------------------------
# quantile_returns
# ----------------------------------------------------------------------
def test_quantile_returns_is_monotone_for_a_signal_that_orders_returns() -> None:
    got = quantile_returns(_predictions())
    means = got.mean()
    assert list(means.index) == ["q1", "q2", "q3", "q4", "q5"]
    assert means.is_monotonic_increasing


def test_quantile_returns_fills_every_bucket_with_the_same_number_of_names() -> None:
    """12 names / 5 buckets -> the bucket sizes may differ by at most one."""
    rows = [(f"S{i}", float(i), float(i)) for i in range(10)]
    got = quantile_returns(_flat_predictions(rows), n_quantiles=5)
    assert got.shape == (1, 5)
    assert got.notna().all().all()


def test_quantile_returns_puts_the_best_prediction_in_the_top_bucket() -> None:
    rows = [(f"S{i}", float(i), float(i)) for i in range(10)]
    got = quantile_returns(_flat_predictions(rows), n_quantiles=5)
    assert got["q5"].iloc[0] > got["q1"].iloc[0]


def test_quantile_returns_is_all_nan_when_a_date_is_too_thin() -> None:
    """A thin date must not have buckets quietly absorbing their neighbours."""
    rows = [(f"S{i}", float(i), float(i)) for i in range(6)]
    got = quantile_returns(_flat_predictions(rows), n_quantiles=5)
    assert got.shape == (1, 5)
    assert got.isna().all().all()


def test_quantile_returns_leaves_a_thin_date_nan_but_keeps_a_thick_one() -> None:
    thick = _flat_predictions([(f"S{i}", float(i), float(i)) for i in range(10)])
    thin = _flat_predictions([(f"T{i}", float(i), float(i)) for i in range(4)], date="2024-01-03")
    got = quantile_returns(pd.concat([thick, thin], ignore_index=True), n_quantiles=5)
    assert got.loc[pd.Timestamp("2024-01-02")].notna().all()
    assert got.loc[pd.Timestamp("2024-01-03")].isna().all()


def test_quantile_returns_is_empty_for_an_empty_input() -> None:
    assert quantile_returns(_predictions().head(0)).empty


def test_quantile_returns_names_its_index() -> None:
    assert quantile_returns(_predictions()).index.name == "date"


# ----------------------------------------------------------------------
# top_quantile_stats
# ----------------------------------------------------------------------
def test_top_quantile_stats_reports_the_expected_keys() -> None:
    got = top_quantile_stats(_predictions())
    assert set(got) == {
        "top_quantile_return",
        "bottom_quantile_return",
        "long_short_spread",
        "long_short_ir",
        "hit_ratio",
    }
    assert got["long_short_spread"] == pytest.approx(
        got["top_quantile_return"] - got["bottom_quantile_return"]
    )
    assert 0.0 <= got["hit_ratio"] <= 1.0


def test_top_quantile_stats_spread_equals_the_quantile_frame() -> None:
    pred = _predictions()
    q = quantile_returns(pred)
    got = top_quantile_stats(pred)
    assert got["top_quantile_return"] == pytest.approx(q["q5"].mean())
    assert got["bottom_quantile_return"] == pytest.approx(q["q1"].mean())


def test_top_quantile_stats_is_empty_for_an_empty_input() -> None:
    assert top_quantile_stats(_predictions().head(0)) == {}


def test_hit_ratio_excludes_dates_without_enough_names() -> None:
    thick = _flat_predictions([(f"S{i}", float(i), float(i)) for i in range(10)])
    thin = _flat_predictions([(f"T{i}", float(i), float(i)) for i in range(4)], date="2024-01-03")
    got = top_quantile_stats(pd.concat([thick, thin], ignore_index=True))
    assert got["hit_ratio"] == pytest.approx(1.0)
    assert np.isnan(top_quantile_stats(thin)["hit_ratio"])


def test_top_quantile_stats_is_nan_when_the_spread_never_moves() -> None:
    rows = [(f"S{i}", float(i), float(i)) for i in range(10)]
    got = top_quantile_stats(_flat_predictions(rows))
    assert np.isnan(got["long_short_ir"])


# ----------------------------------------------------------------------
# prediction_turnover
# ----------------------------------------------------------------------
def _turnover_frame(preds: dict[str, dict[str, float]]) -> pd.DataFrame:
    frames = []
    for date, values in preds.items():
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.Timestamp(date),
                    "symbol": list(values),
                    "prediction": list(values.values()),
                    "realised": 0.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_prediction_turnover_matches_a_hand_computed_value() -> None:
    """10 names, top 20% = 2 names: {S0,S1} -> {S0,S1} -> {S8,S9} averages to 0.5."""
    ranking = {f"S{i}": float(10 - i) for i in range(10)}  # S0=10 ... S9=1
    rotated = {f"S{i}": float(10 - (i + 2) % 10) for i in range(10)}  # S8=10, S9=9
    frame = _turnover_frame({"2024-01-02": ranking, "2024-01-03": ranking, "2024-01-04": rotated})
    assert prediction_turnover(frame) == pytest.approx(0.5)


def test_prediction_turnover_is_zero_when_the_book_never_changes() -> None:
    ranking = {f"S{i}": float(10 - i) for i in range(10)}
    frame = _turnover_frame({"2024-01-02": ranking, "2024-01-03": ranking, "2024-01-04": ranking})
    assert prediction_turnover(frame) == pytest.approx(0.0)


def test_prediction_turnover_is_one_when_the_book_rotates_completely() -> None:
    a = {f"A{i}": float(10 - i) for i in range(10)}
    b = {f"B{i}": float(10 - i) for i in range(10)}
    frame = _turnover_frame({"2024-01-02": a, "2024-01-03": b})
    assert prediction_turnover(frame) == pytest.approx(1.0)


def test_prediction_turnover_skips_a_date_with_too_few_names() -> None:
    ranking = {f"S{i}": float(10 - i) for i in range(10)}
    thin = {f"T{i}": float(3 - i) for i in range(3)}
    frame = _turnover_frame({"2024-01-02": ranking, "2024-01-03": thin, "2024-01-04": ranking})
    assert prediction_turnover(frame) == pytest.approx(0.0)


def test_prediction_turnover_is_nan_without_two_usable_dates() -> None:
    assert np.isnan(prediction_turnover(_predictions().head(0)))
    ranking = {f"S{i}": float(10 - i) for i in range(10)}
    assert np.isnan(prediction_turnover(_turnover_frame({"2024-01-02": ranking})))


# ----------------------------------------------------------------------
# evaluate_predictions
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def evaluation() -> ModelEvaluation:
    return evaluate_predictions(_predictions(), model_name="ridge", horizon=21)


def test_evaluate_reports_the_expected_summary_keys(evaluation: ModelEvaluation) -> None:
    assert set(evaluation.summary) >= {
        "model",
        "rank_ic_mean",
        "rank_ic_std",
        "icir",
        "t_stat",
        "positive_ic_ratio",
        "n_periods",
        "turnover",
        "ann_long_short",
        "long_short_ir",
        "hit_ratio",
    }


def test_evaluate_rank_ic_mean_is_the_mean_of_the_daily_series(evaluation: ModelEvaluation) -> None:
    clean = evaluation.ic_series.dropna()
    assert evaluation.summary["rank_ic_mean"] == pytest.approx(clean.mean())
    assert evaluation.summary["rank_ic_std"] == pytest.approx(clean.std())
    assert evaluation.summary["n_periods"] == len(clean)


def test_evaluate_applies_the_overlap_correction_to_the_t_stat(evaluation: ModelEvaluation) -> None:
    """``n / horizon`` effective observations, not a naive ``sqrt(n)``."""
    clean = evaluation.ic_series.dropna()
    eff_n = max(len(clean) / 21, 2.0)
    assert evaluation.summary["t_stat"] == pytest.approx(
        clean.mean() / clean.std() * np.sqrt(eff_n)
    )
    naive = clean.mean() / clean.std() * np.sqrt(len(clean))
    assert evaluation.summary["t_stat"] < naive


def test_evaluate_annualises_the_long_short_spread(evaluation: ModelEvaluation) -> None:
    assert evaluation.summary["ann_long_short"] == pytest.approx(
        evaluation.summary["long_short_spread"] * 252.0 / 21.0
    )


def test_evaluate_positive_ic_ratio_is_a_fraction(evaluation: ModelEvaluation) -> None:
    clean = evaluation.ic_series.dropna()
    assert evaluation.summary["positive_ic_ratio"] == pytest.approx((clean > 0).mean())


def test_evaluate_carries_the_inputs_through(evaluation: ModelEvaluation) -> None:
    assert evaluation.model_name == "ridge"
    assert not evaluation.predictions.empty
    assert evaluation.quantile_returns.shape[1] == 5
    assert evaluation.to_dict() == {"model": "ridge", **evaluation.summary}


def test_evaluate_builds_one_yearly_row_per_calendar_year(evaluation: ModelEvaluation) -> None:
    yearly = evaluation.yearly_metrics
    assert list(yearly["year"]) == sorted(evaluation.ic_series.dropna().index.year.unique())
    assert set(yearly.columns) >= {"year", "rank_ic_mean", "rank_ic_std", "rank_ic_count", "icir"}
    for _, row in yearly.iterrows():
        assert row["icir"] == pytest.approx(row["rank_ic_mean"] / row["rank_ic_std"])


def test_evaluate_yearly_icir_is_nan_for_a_single_year() -> None:
    """One IC in a year has no standard deviation - and must not divide by zero."""
    single = _predictions()
    one_date = single["date"].iloc[0]
    got = evaluate_predictions(single[single["date"] == one_date], model_name="one")
    assert np.isnan(got.yearly_metrics["icir"].iloc[0])


def test_evaluate_has_no_fold_metrics_without_a_fold_id(evaluation: ModelEvaluation) -> None:
    assert evaluation.fold_metrics.empty


def test_evaluate_builds_one_fold_row_per_fold(evaluation: ModelEvaluation) -> None:
    pred = _predictions()
    fold_id = pd.Series(np.repeat(np.arange(4), len(pred) // 4), index=pred.index)
    folds = evaluate_predictions(pred, model_name="ridge", fold_id=fold_id).fold_metrics
    assert list(folds["fold"]) == [0, 1, 2, 3]
    assert set(folds.columns) >= {"fold", "rank_ic_mean", "rank_ic_std", "n_days", "icir"}


def test_fold_metrics_carries_exactly_one_n_days_column() -> None:
    """Regression: ``n_obs`` used to be renamed onto the existing ``n_days``.

    The frame then held two columns named ``n_days`` and ``folds["n_days"]``
    returned a DataFrame, while the surviving value was the prediction-row count
    (n_dates x n_symbols) rather than the number of IC observations.
    """
    pred = _predictions()
    fold_id = pd.Series(np.repeat(np.arange(4), len(pred) // 4), index=pred.index)
    folds = evaluate_predictions(pred, model_name="ridge", fold_id=fold_id).fold_metrics
    assert list(folds.columns).count("n_days") == 1
    assert isinstance(folds["n_days"], pd.Series)
    assert folds["n_days"].tolist() == [len(DATES) // 4] * 4


def test_evaluate_fold_icir_is_nan_for_a_single_observation() -> None:
    pred = _predictions()
    fold_id = pd.Series(0, index=pred.index)
    fold_id.iloc[1:] = 1
    folds = evaluate_predictions(pred, model_name="ridge", fold_id=fold_id).fold_metrics
    assert np.isnan(folds.loc[folds["fold"] == 0, "icir"]).all()


def test_evaluate_survives_a_constant_realised_return() -> None:
    """No dispersion -> no IC -> NaN metrics, not a crash and not a zero."""
    flat = _predictions()
    flat["realised"] = 0.0
    got = evaluate_predictions(flat, model_name="flat")
    assert got.summary["n_periods"] == 0
    assert np.isnan(got.summary["rank_ic_mean"])
    assert np.isnan(got.summary["icir"])
    assert np.isnan(got.summary["t_stat"])
    assert np.isnan(got.summary["positive_ic_ratio"])
    assert got.yearly_metrics.empty


def test_evaluate_respects_periods_per_year(evaluation: ModelEvaluation) -> None:
    pred = _predictions()
    got = evaluate_predictions(pred, model_name="ridge", horizon=21, periods_per_year=52.0)
    assert got.summary["ann_long_short"] == pytest.approx(
        got.summary["long_short_spread"] * 52.0 / 21.0
    )


def test_evaluate_ic_series_matches_daily_rank_ic(evaluation: ModelEvaluation) -> None:
    np.testing.assert_allclose(
        evaluation.ic_series.to_numpy(dtype=float),
        daily_rank_ic(_predictions()).to_numpy(dtype=float),
        atol=1e-12,
    )


# ----------------------------------------------------------------------
# compare_models
# ----------------------------------------------------------------------
def test_compare_models_sorts_by_rank_ic_descending(evaluation: ModelEvaluation) -> None:
    weak = evaluate_predictions(_predictions(noise=20.0, seed=5), model_name="weak")
    table = compare_models({"strong": evaluation, "weak": weak})
    assert list(table["model"]) == ["ridge", "weak"]
    assert table["rank_ic_mean"].is_monotonic_decreasing


def test_compare_models_keeps_the_expected_columns(evaluation: ModelEvaluation) -> None:
    table = compare_models({"strong": evaluation})
    assert set(table.columns) <= {
        "model",
        "rank_ic_mean",
        "icir",
        "rank_icir",
        "t_stat",
        "turnover",
        "ann_long_short",
        "long_short_ir",
        "hit_ratio",
    }
    assert "model" in table.columns
    assert "rank_ic_mean" in table.columns
    assert list(table.index) == list(range(len(table)))


def test_compare_models_returns_an_empty_frame_for_no_evaluations() -> None:
    assert compare_models({}).empty

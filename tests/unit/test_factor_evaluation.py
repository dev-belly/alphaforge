"""Unit tests for the factor evaluation layer (IC, quantiles, decay, FDR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.factors.evaluation import (
    benjamini_hochberg,
    compute_ic,
    factor_turnover,
    ic_decay,
    quantile_portfolios,
)


def _synth_factor_close(
    n: int = 400, k: int = 10, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A factor that is *predictive* of next-period returns + the close panel."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    syms = [f"S{i}" for i in range(k)]
    f = pd.DataFrame(rng.normal(0, 1, size=(n, k)), index=idx, columns=syms)
    # Forward return is weakly driven by the factor (positive IC) + noise.
    fwd = f * 0.02 + rng.normal(0, 0.02, size=(n, k))
    close = pd.DataFrame((1.0 + fwd.fillna(0.0)).cumprod(axis=0) * 100.0, index=idx, columns=syms)
    return f, close


def test_compute_ic_positive_and_finite():
    f, close = _synth_factor_close(seed=3)
    fwd = close.shift(-21) / close - 1.0
    ic = compute_ic(f, fwd)
    assert "pearson_ic" in ic.columns and "rank_ic" in ic.columns
    valid = ic.dropna()
    assert len(valid) >= 5
    assert valid["rank_ic"].std() > 0
    # The engineered factor has positive mean rank IC.
    assert valid["rank_ic"].mean() > 0


def test_compute_ic_respects_universe():
    f, close = _synth_factor_close(seed=4)
    fwd = close.shift(-21) / close - 1.0
    universe = pd.DataFrame(True, index=f.index, columns=f.columns)
    universe.iloc[:, :5] = False  # restrict to half the names
    ic_full = compute_ic(f, fwd)
    ic_univ = compute_ic(f, fwd, universe=universe)
    assert (ic_univ.notna().sum(axis=1) <= ic_full.notna().sum(axis=1) + 1).all()


@pytest.mark.parametrize(
    ("scores", "outcomes"),
    [
        ([0, 10, 20, 30, 40, np.nan, 15], [0.1, 0.3, 0.2, 0.5, 0.4, 0.25, np.nan]),
        ([1, 1, 2, 3, 3, 2, np.nan, 4], [2, 3, 3, 1, 4, np.nan, 2, 4]),
        ([0, 10, 20, 30, 40, np.inf, 15], [0.1, 0.3, 0.2, 0.5, 0.4, 0.25, -np.inf]),
        ([2, 2, 2, 2, 2, 3, np.nan], [1, 2, 3, 4, 5, np.nan, 6]),
    ],
    ids=["missing-pairs", "ties", "nonfinite-pairs", "constant-shared-scores"],
)
def test_compute_ic_matches_correlations_of_shared_finite_pairs(scores, outcomes):
    dates = pd.date_range("2024-01-02", periods=1)
    factor = pd.DataFrame([scores], index=dates)
    forward = pd.DataFrame([outcomes], index=dates)
    paired = pd.DataFrame({"factor": scores, "forward": outcomes})
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna()

    actual = compute_ic(factor, forward).iloc[0]

    assert actual["n"] == len(paired)
    for method, column in [("pearson", "pearson_ic"), ("spearman", "rank_ic")]:
        expected = paired["factor"].corr(paired["forward"], method=method)
        if pd.isna(expected):
            assert pd.isna(actual[column])
        else:
            assert actual[column] == pytest.approx(expected)


def test_compute_ic_aligns_labels_and_applies_universe_before_ranking():
    dates = pd.date_range("2024-01-02", periods=2)
    columns = list("ABCDEFGH")
    factor = pd.DataFrame([[0, 10, 20, 30, 40, 15, 35, 25]] * 2, index=dates, columns=columns)
    forward = pd.DataFrame(
        [[0.1, 0.3, 0.2, 0.5, 0.4, np.nan, 0.35, 0.25]] * 2,
        index=dates,
        columns=columns,
    )
    universe = pd.DataFrame(True, index=dates, columns=columns)
    universe.loc[:, ["G", "H"]] = False
    universe.loc[dates[1], "E"] = False

    actual = compute_ic(factor.iloc[::-1, ::-1], forward, universe.iloc[::-1, ::-1])

    assert actual.loc[dates[0], "n"] == 5
    assert actual.loc[dates[0], "rank_ic"] == pytest.approx(0.8)
    assert actual.loc[dates[1], "n"] == 4
    assert actual.loc[dates[1], ["pearson_ic", "rank_ic"]].isna().all()


def test_quantile_portfolios_spread_positive():
    f, close = _synth_factor_close(seed=5)
    fwd = close.shift(-21) / close - 1.0
    qrets, top = quantile_portfolios(f, fwd, n_quantiles=5)
    assert list(qrets.columns) == ["q1", "q2", "q3", "q4", "q5"]
    # Top quantile mean return should beat the bottom quantile on average.
    assert qrets["q5"].mean() > qrets["q1"].mean()


@pytest.mark.parametrize(("n_assets", "n_quantiles"), [(10, 5), (20, 5), (12, 3)])
@pytest.mark.parametrize("tied", [False, True], ids=["distinct", "tied"])
def test_quantile_portfolios_have_equal_sized_sorted_groups(n_assets, n_quantiles, tied):
    dates = pd.date_range("2024-01-02", periods=1)
    scores = np.zeros(n_assets) if tied else np.arange(n_assets)
    factor = pd.DataFrame([scores], index=dates)
    outcomes = np.arange(n_assets) ** 2 / 1000.0
    forward = pd.DataFrame([outcomes], index=dates)

    actual, top = quantile_portfolios(factor, forward, n_quantiles=n_quantiles)

    # Hand-partition the sorted names: ten stocks into five groups means two
    # names in every group, including both endpoints. Ties retain column order.
    groups = np.split(outcomes, n_quantiles)
    assert actual.iloc[0].to_numpy() == pytest.approx([group.mean() for group in groups])
    expected_top = list(range(n_assets - n_assets // n_quantiles, n_assets))
    assert list(top.columns[top.iloc[0]]) == expected_top


def test_quantile_portfolios_ignore_unpaired_and_nonfinite_names_before_bucketing():
    dates = pd.date_range("2024-01-02", periods=1)
    columns = [f"S{i}" for i in range(10)]
    paired_factor = pd.DataFrame([np.arange(10)], index=dates, columns=columns)
    paired_forward = paired_factor / 100.0
    factor = paired_factor.assign(missing_outcome=4.5, missing_factor=np.nan, infinite=5.5)
    forward = paired_forward.assign(missing_outcome=np.nan, missing_factor=0.06, infinite=np.inf)

    actual, top = quantile_portfolios(factor.iloc[:, ::-1], forward)
    expected, expected_top = quantile_portfolios(paired_factor, paired_forward)

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(top[columns], expected_top)
    assert not top[["missing_outcome", "missing_factor", "infinite"]].any().any()


@pytest.mark.parametrize("n_assets", [11, 13])
def test_quantile_portfolios_partition_uneven_cross_sections_without_losing_names(n_assets):
    dates = pd.date_range("2024-01-02", periods=n_assets)
    factor = pd.DataFrame(np.tile(np.arange(n_assets), (n_assets, 1)), index=dates)
    # Each date probes one name, so positive group means identify membership
    # independently of the implementation's rank-to-bucket calculation.
    forward = pd.DataFrame(np.eye(n_assets), index=dates)

    actual, top = quantile_portfolios(factor, forward)
    membership = actual > 0
    counts = membership.sum(axis=0)

    assert (membership.sum(axis=1) == 1).all()
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(top.iloc[0].to_numpy(), membership["q5"].to_numpy())


def test_quantile_portfolios_share_universe_and_minimum_pair_count():
    dates = pd.date_range("2024-01-02", periods=2)
    factor = pd.DataFrame([np.arange(12)] * 2, index=dates)
    forward = factor / 100.0
    universe = pd.DataFrame(True, index=dates, columns=factor.columns)
    universe.loc[:, [10, 11]] = False
    forward.loc[dates[1], 9] = np.nan

    actual, top = quantile_portfolios(factor, forward, universe=universe)

    assert actual.iloc[0].to_numpy() == pytest.approx([0.005, 0.025, 0.045, 0.065, 0.085])
    assert list(top.columns[top.iloc[0]]) == [8, 9]
    assert actual.iloc[1].isna().all()
    assert not top.iloc[1].any()


def test_factor_turnover_in_unit_interval():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-01-01", periods=100, freq="B")
    w = pd.DataFrame(rng.uniform(0, 1, size=(100, 8)), index=idx, columns=list("ABCDEFGH"))
    to = factor_turnover(w)
    assert ((to >= 0) & (to <= 2.0)).all()  # one-way turnover <= 2 by construction


def test_ic_decay_has_expected_horizons():
    f, close = _synth_factor_close(seed=8)
    decay = ic_decay(f, close, horizons=[1, 5, 21, 63])
    assert list(decay.index) == [1, 5, 21, 63]
    assert "rank_ic_mean" in decay.columns


def test_benjamini_hochberg_controls_fdr():
    # All-null p-values => nothing significant.
    pvals = pd.Series({f"f{i}": np.nan for i in range(5)})
    assert not benjamini_hochberg(pvals).any()
    # A mixture: tiny p (sig) and 1.0 (ns).
    p = pd.Series({"a": 0.001, "b": 0.5, "c": 0.9, "d": 0.001, "e": 0.8})
    passed = benjamini_hochberg(p, alpha=0.05)
    assert passed["a"] and passed["d"]
    assert not passed["b"] and not passed["c"]

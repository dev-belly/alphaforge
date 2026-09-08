"""Offline tests for the cross-sectional factor preprocessing chain.

Every factor that reaches portfolio construction passes through
``FactorPreprocessor.process``, so a defect here silently corrupts *all* factors
and every downstream backtest - and it stays invisible in a smoke test, which
only proves the pipeline runs. These tests lock the actual mathematics:

  * winsorization / z-scoring / ranking happen **within a single date** and never
    leak information across dates;
  * industry + size neutralisation follows Frisch-Waugh-Lovell, so the residual
    must be orthogonal to the industry dummies *and* to within-industry-demeaned
    log market cap;
  * a name the model cannot score must receive the neutral ``fill_value`` -
    never a fabricated score.

Everything is deterministic: no network, no randomness, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.factors.base import Factor, FactorContext, FactorSpec, MarketPanel
from alphaforge.factors.preprocessing import (
    FactorPreprocessor,
    ProcessingConfig,
    demean_panel,
    group_mean,
    neutralize_continuous,
    rank_panel,
    standardize_panel,
    winsorize_panel,
)

DATES = pd.bdate_range("2024-01-02", periods=6)
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
# 2 Energy names + 4 Tech names - deliberately unbalanced.
INDUSTRIES = ["Energy", "Energy", "Tech", "Tech", "Tech", "Tech"]
ENERGY = ["AAA", "BBB"]
TECH = ["CCC", "DDD", "EEE", "FFF"]


def _panel(
    *,
    industry: pd.DataFrame | None = None,
    market_cap: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
) -> MarketPanel:
    close = pd.DataFrame(
        {s: np.linspace(100.0, 100.0 + i * 10.0, len(DATES)) for i, s in enumerate(SYMBOLS)},
        index=DATES,
    )
    if market_cap is None:
        market_cap = pd.DataFrame(
            {s: 1.0e9 * (i + 1) for i, s in enumerate(SYMBOLS)},
            index=DATES,
            columns=SYMBOLS,
        )
    if universe is None:
        universe = pd.DataFrame(True, index=DATES, columns=SYMBOLS)
    if industry is None:
        industry = pd.DataFrame([INDUSTRIES] * len(DATES), index=DATES, columns=SYMBOLS)
    return MarketPanel(
        dates=DATES,
        close=close,
        raw_close=close,
        returns=close.pct_change(fill_method=None),
        volume=pd.DataFrame(1.0e6, index=DATES, columns=SYMBOLS),
        dollar_volume=pd.DataFrame(1.0e8, index=DATES, columns=SYMBOLS),
        market_cap=market_cap,
        universe=universe,
        industry=industry,
    )


def _raw(value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(value, index=DATES, columns=SYMBOLS)


def _varying_raw() -> pd.DataFrame:
    """A non-degenerate panel: every row varies across symbols.

    The non-zero starting level matters. A first row of all zeros is *flat*,
    which makes z-scoring, ranking and neutralisation degenerate - a real defect
    then hides behind a row of zeros.
    """

    return pd.DataFrame(
        {
            s: 10.0 + i * 3.0 + np.arange(len(DATES), dtype=float) * 0.5
            for i, s in enumerate(SYMBOLS)
        },
        index=DATES,
    )


def _factor(raw: pd.DataFrame, direction: int = 1) -> Factor:
    return Factor(spec=FactorSpec(name="test", category="momentum", direction=direction), raw=raw)


def _config(**overrides: object) -> ProcessingConfig:
    """Every transform off, so each test switches on exactly one behaviour."""
    base: dict[str, object] = dict(
        winsorize=False,
        standardize=False,
        rank_transform=False,
        industry_neutralize=False,
        size_neutralize=False,
        demean=False,
    )
    base.update(overrides)
    return ProcessingConfig(**base)


# --------------------------------------------------------------------------
# ProcessingConfig
# --------------------------------------------------------------------------


def test_from_dict_keeps_known_keys_and_ignores_the_rest() -> None:
    cfg = ProcessingConfig.from_dict({"winsorize": False, "min_names": 3, "typo": 1})
    assert cfg.winsorize is False
    assert cfg.min_names == 3
    assert cfg.standardize is True  # untouched defaults survive


def test_from_dict_treats_none_as_empty() -> None:
    assert ProcessingConfig.from_dict(None) == ProcessingConfig()


# --------------------------------------------------------------------------
# group_mean
# --------------------------------------------------------------------------


def test_group_mean_broadcasts_the_industry_mean() -> None:
    df = pd.DataFrame([[1.0, 3.0, 10.0, 20.0, 30.0, 40.0]], index=DATES[:1], columns=SYMBOLS)
    out = group_mean(df, pd.Series(INDUSTRIES, index=SYMBOLS))
    # Energy mean = 2.0 ; Tech mean = 25.0
    np.testing.assert_allclose(out.iloc[0], [2.0, 2.0, 25.0, 25.0, 25.0, 25.0])


def test_group_mean_falls_back_to_the_grand_mean_without_labels() -> None:
    df = pd.DataFrame([[1.0, 3.0, 10.0, 20.0]], index=DATES[:1], columns=SYMBOLS[:4])
    mapping = pd.Series([np.nan] * 4, index=SYMBOLS[:4])
    np.testing.assert_allclose(group_mean(df, mapping).iloc[0], [8.5] * 4)


def test_group_mean_leaves_unlabelled_names_nan() -> None:
    """A name with no industry cannot be industry-adjusted, so it stays NaN.

    Note the asymmetry worth knowing about: *no* labels at all falls back to the
    grand mean, but a *partial* mapping leaves the unlabelled names unadjusted.
    """
    df = pd.DataFrame([[1.0, 3.0, 10.0, 20.0]], index=DATES[:1], columns=SYMBOLS[:4])
    mapping = pd.Series(["Energy", "Energy", np.nan, np.nan], index=SYMBOLS[:4])
    out = group_mean(df, mapping)
    assert out.iloc[0, 0] == pytest.approx(2.0)
    assert np.isnan(out.iloc[0, 2])


# --------------------------------------------------------------------------
# Panel transforms
# --------------------------------------------------------------------------


def test_winsorize_clips_each_date_to_its_own_quantiles() -> None:
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0, 100.0], [1000.0, 2000.0, 3000.0, 4000.0]],
        index=DATES[:2],
        columns=SYMBOLS[:4],
    )
    out = winsorize_panel(df, 0.25)
    for row in out.index:
        assert out.loc[row].min() >= df.loc[row].quantile(0.25) - 1e-12
        assert out.loc[row].max() <= df.loc[row].quantile(0.75) + 1e-12


@pytest.mark.parametrize("lower", [0.0, -0.1])
def test_winsorize_is_a_noop_when_disabled(lower: float) -> None:
    df = pd.DataFrame([[1.0, 2.0, 3.0, 100.0]], index=DATES[:1], columns=SYMBOLS[:4])
    pd.testing.assert_frame_equal(winsorize_panel(df, lower), df)


def test_standardize_is_a_nan_preserving_zscore() -> None:
    df = pd.DataFrame([[1.0, 2.0, 3.0, np.nan]], index=DATES[:1], columns=SYMBOLS[:4])
    out = standardize_panel(df)
    assert out.iloc[0].mean() == pytest.approx(0.0)
    assert out.iloc[0].std(ddof=1) == pytest.approx(1.0)
    assert np.isnan(out.iloc[0, 3])  # NaN in -> NaN out, never a fabricated 0


def test_standardize_yields_nan_not_inf_when_the_cross_section_is_constant() -> None:
    df = pd.DataFrame([[5.0] * 4], index=DATES[:1], columns=SYMBOLS[:4])
    out = standardize_panel(df)
    assert out.isna().all().all()


def test_rank_panel_maps_to_percentiles() -> None:
    df = pd.DataFrame([[10.0, 20.0, 30.0, np.nan]], index=DATES[:1], columns=SYMBOLS[:4])
    out = rank_panel(df)
    assert out.iloc[0, 0] == pytest.approx(1.0 / 3.0)
    assert out.iloc[0, 2] == pytest.approx(1.0)
    assert np.isnan(out.iloc[0, 3])


def test_demean_panel_removes_the_row_mean() -> None:
    df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], index=DATES[:2])
    np.testing.assert_allclose(demean_panel(df).mean(axis=1), 0.0, atol=1e-12)


def test_neutralize_continuous_residual_is_orthogonal_to_the_control() -> None:
    y = pd.DataFrame([[1.0, 2.0, 5.0, 9.0, 4.0, 7.0]], index=DATES[:1], columns=SYMBOLS)
    s = pd.DataFrame([[1.0, 2.0, 3.0, 8.0, 5.0, 6.0]], index=DATES[:1], columns=SYMBOLS)
    out = neutralize_continuous(y, s)
    sm = s.sub(s.mean(axis=1), axis=0)
    np.testing.assert_allclose((out * sm).sum(axis=1), 0.0, atol=1e-10)


def test_neutralize_continuous_leaves_y_demeaned_when_the_control_is_flat() -> None:
    """A constant control carries no information, so nothing is removed."""
    y = pd.DataFrame([[1.0, 2.0, 5.0, 9.0]], index=DATES[:1], columns=SYMBOLS[:4])
    flat = pd.DataFrame([[3.0] * 4], index=DATES[:1], columns=SYMBOLS[:4])
    out = neutralize_continuous(y, flat)
    np.testing.assert_allclose(out, y.sub(y.mean(axis=1), axis=0))


# --------------------------------------------------------------------------
# FactorPreprocessor wiring
# --------------------------------------------------------------------------


def test_preprocessor_takes_the_latest_industry_and_log_size() -> None:
    panel = _panel()
    pre = FactorPreprocessor(_config(), FactorContext(panel))
    assert pre._industry.tolist() == INDUSTRIES
    np.testing.assert_allclose(pre._size.to_numpy(), np.log(panel.market_cap.to_numpy()))


def test_preprocessor_tolerates_an_empty_industry_panel() -> None:
    pre = FactorPreprocessor(_config(), FactorContext(_panel(industry=pd.DataFrame())))
    assert len(pre._industry) == 0


# --------------------------------------------------------------------------
# process()
# --------------------------------------------------------------------------


def test_process_masks_names_outside_the_investable_universe() -> None:
    universe = pd.DataFrame(True, index=DATES, columns=SYMBOLS)
    universe["FFF"] = False
    out = FactorPreprocessor(_config(), FactorContext(_panel(universe=universe))).process(
        _factor(_raw())
    )
    assert (out["FFF"] == 0.0).all()  # excluded -> neutral fill
    assert (out["AAA"] == 1.0).all()  # investable -> keeps its score


def test_process_can_skip_the_universe_mask() -> None:
    universe = pd.DataFrame(True, index=DATES, columns=SYMBOLS)
    universe["FFF"] = False
    out = FactorPreprocessor(_config(), FactorContext(_panel(universe=universe))).process(
        _factor(_raw()), apply_universe=False
    )
    assert (out["FFF"] == 1.0).all()


def test_process_drops_dates_with_too_few_scored_names() -> None:
    raw = _raw()
    raw.iloc[0, 3:] = np.nan  # only 3 names score on the first date
    out = FactorPreprocessor(_config(min_names=5), FactorContext(_panel())).process(_factor(raw))
    assert out.iloc[0].isna().all()
    assert out.iloc[1].notna().all()


def test_process_orients_a_negative_factor() -> None:
    raw = pd.DataFrame(
        {s: np.arange(len(DATES), dtype=float) + i for i, s in enumerate(SYMBOLS)}, index=DATES
    )
    out = FactorPreprocessor(_config(), FactorContext(_panel())).process(_factor(raw, direction=-1))
    np.testing.assert_allclose(out.to_numpy(), -raw.to_numpy())


def test_process_winsorizes_each_date() -> None:
    raw = _raw()
    raw.iloc[:, 0] = 1000.0
    pre = FactorPreprocessor(_config(winsorize=True, winsorization=0.2), FactorContext(_panel()))
    out = pre.process(_factor(raw))
    assert out.abs().max().max() <= 1.0 + 1e-12


def test_process_removes_the_industry_mean() -> None:
    raw = _varying_raw()
    pre = FactorPreprocessor(_config(industry_neutralize=True), FactorContext(_panel()))
    out = pre.process(_factor(raw))
    for members in (ENERGY, TECH):
        np.testing.assert_allclose(out[members].mean(axis=1), 0.0, atol=1e-12)


def test_process_removes_size_exposure() -> None:
    raw = _varying_raw()
    pre = FactorPreprocessor(_config(size_neutralize=True), FactorContext(_panel()))
    out = pre.process(_factor(raw))
    log_cap = np.log(_panel().market_cap)
    sm = log_cap.sub(log_cap.mean(axis=1), axis=0)
    np.testing.assert_allclose((out * sm).sum(axis=1), 0.0, atol=1e-9)


def test_process_industry_and_size_neutralisation_is_joint() -> None:
    """Frisch-Waugh-Lovell: the residual is orthogonal to *both* controls."""
    raw = _varying_raw()
    panel = _panel()
    pre = FactorPreprocessor(
        _config(industry_neutralize=True, size_neutralize=True), FactorContext(panel)
    )
    out = pre.process(_factor(raw))

    log_cap = np.log(panel.market_cap)
    size_perp = log_cap - group_mean(log_cap, pd.Series(INDUSTRIES, index=SYMBOLS))
    np.testing.assert_allclose((out * size_perp).sum(axis=1), 0.0, atol=1e-9)
    for members in (ENERGY, TECH):
        np.testing.assert_allclose(out[members].mean(axis=1), 0.0, atol=1e-9)


def test_process_demeans_when_no_neutralisation_is_asked_for() -> None:
    raw = _varying_raw()
    pre = FactorPreprocessor(_config(demean=True), FactorContext(_panel()))
    out = pre.process(_factor(raw))
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)


def test_process_ranks_before_standardising() -> None:
    raw = _varying_raw()
    pre = FactorPreprocessor(
        _config(rank_transform=True, standardize=True), FactorContext(_panel())
    )
    out = pre.process(_factor(raw))
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=1, ddof=1), 1.0, atol=1e-12)
    assert out.iloc[1].idxmax() == raw.iloc[1].idxmax()  # ranking preserves order


def test_process_gives_unscored_names_the_neutral_fill_value() -> None:
    """A name with no raw factor value must not be handed a fabricated score.

    The residualisation uses ``sub(..., fill_value=0.0)``, which would otherwise
    turn a missing factor value into ``-beta * size`` - a real-looking score for
    a name the model never scored at all.
    """
    raw = _varying_raw()
    raw["FFF"] = np.nan
    pre = FactorPreprocessor(_config(size_neutralize=True), FactorContext(_panel()))
    out = pre.process(_factor(raw))
    np.testing.assert_allclose(out["FFF"].to_numpy(), 0.0, atol=1e-12)


def test_process_gives_unscored_names_the_neutral_fill_value_when_joint() -> None:
    raw = _varying_raw()
    raw["FFF"] = np.nan
    pre = FactorPreprocessor(
        _config(industry_neutralize=True, size_neutralize=True), FactorContext(_panel())
    )
    out = pre.process(_factor(raw))
    np.testing.assert_allclose(out["FFF"].to_numpy(), 0.0, atol=1e-12)


def test_process_many_processes_every_factor() -> None:
    raw = _varying_raw()
    factors = {"a": _factor(raw), "b": _factor(raw * 2.0)}
    out = FactorPreprocessor(_config(demean=True), FactorContext(_panel())).process_many(factors)
    assert set(out) == {"a", "b"}
    # Demeaning is linear, so scaling the input scales the output.
    np.testing.assert_allclose(out["b"].to_numpy(), 2.0 * out["a"].to_numpy())

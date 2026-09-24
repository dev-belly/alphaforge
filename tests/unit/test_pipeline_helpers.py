"""Offline tests for the research pipeline's pure helpers.

``ResearchPipeline.run`` is the 200-line orchestrator that the integration suite
already drives end to end against a real run, so the value here is in the pieces
around it - the ones that quietly decide *what the risk model sees* and *what the
report claims*:

  * ``_style_factor_exposures`` maps the risk model's style factors onto
    factor-library panels by an ordered alias list. Pick the wrong alias, or take
    the wrong axis of the panel, and the risk model is handed a
    time-series-shaped object where it expects a cross-sectional one - which the
    optimiser will happily consume.
  * ``_decomp_table`` attaches the covariance the chart needs, and falls back to
    the factor covariance when the decomposition is not available. A fallback
    that silently returns the wrong frame is worse than an error.
  * ``_notes`` and ``as_tool_state`` decide what the briefing and the copilot
    tool layer are told, so a missing or misnamed key shows up as a claim about
    the run rather than as a crash.

Everything is deterministic: no network, no pipeline run, no shared state.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphaforge.pipeline import (
    ResearchPipeline,
    ResearchState,
    _decomp_table,
    _style_factor_exposures,
    run_research,
)
from alphaforge.utils.config import Config

DATES = pd.bdate_range("2024-01-02", periods=12)
SYMBOLS = pd.Index([f"S{i}" for i in range(6)])


def _panel(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(DATES), len(SYMBOLS))), index=DATES, columns=SYMBOLS)


def _risk_result(n_factors: int = 3, n_assets: int = len(SYMBOLS)) -> SimpleNamespace:
    names = [f"F{i}" for i in range(n_factors)]
    rng = np.random.default_rng(5)
    a = rng.normal(size=(n_factors, n_factors))
    factor_cov = pd.DataFrame(a @ a.T, index=names, columns=names)
    exposures = pd.DataFrame(
        rng.normal(size=(n_assets, n_factors)), index=SYMBOLS[:n_assets], columns=names
    )
    specific = pd.Series(rng.uniform(1e-4, 4e-4, size=n_assets), index=SYMBOLS[:n_assets])
    cov = pd.DataFrame(
        np.eye(n_assets) * 1e-3, index=SYMBOLS[:n_assets], columns=SYMBOLS[:n_assets]
    )
    return SimpleNamespace(
        factor_cov=factor_cov,
        covariance=cov,
        exposures=exposures,
        specific_var=specific,
    )


# ----------------------------------------------------------------------
# ResearchState
# ----------------------------------------------------------------------
def test_as_tool_state_exposes_exactly_the_keys_the_tool_layer_reads() -> None:
    state = ResearchState()
    got = state.as_tool_state()
    assert set(got) == {
        "library",
        "model_eval",
        "backtest",
        "weights",
        "risk_result",
        "brinson",
        "factor_attr",
        "regime",
        "stress",
        "config",
        "quality",
    }


def test_as_tool_state_omits_the_heavy_internals() -> None:
    """The panel and the dataset are not inputs to any tool, and are large."""
    got = ResearchState().as_tool_state()
    for internal in ("panel", "dataset", "signal_panel", "report_path", "briefing"):
        assert internal not in got


def test_as_tool_state_passes_the_values_through_by_identity() -> None:
    backtest = object()
    got = ResearchState(backtest=backtest, config={"a": 1}).as_tool_state()
    assert got["backtest"] is backtest
    assert got["config"] == {"a": 1}


# ----------------------------------------------------------------------
# ResearchPipeline construction
# ----------------------------------------------------------------------
def test_a_config_object_is_used_as_is() -> None:
    supplied = Config(raw={"project": {"seed": 7}})
    assert ResearchPipeline(supplied).config is supplied


def test_a_plain_dict_becomes_a_config() -> None:
    got = ResearchPipeline({"project": {"seed": 7}})
    assert isinstance(got.config, Config)
    assert got.config.get("project.seed") == 7


def test_no_config_loads_the_defaults() -> None:
    got = ResearchPipeline(None)
    assert isinstance(got.config, Config)
    assert got.config.get("data.provider")


def test_a_dict_is_copied_at_the_top_level() -> None:
    """``dict(config)`` is a shallow copy: the mapping is new, nested dicts are shared.

    Worth pinning because it is the kind of thing a caller assumes wrongly -
    mutating a *nested* section after construction does reach the pipeline.
    """
    supplied = {"project": {"seed": 7}}
    pipeline = ResearchPipeline(supplied)
    supplied["project"] = {"seed": 999}
    assert pipeline.config.get("project.seed") == 7  # rebinding the key is safe

    shared = {"project": {"seed": 7}}
    other = ResearchPipeline(shared)
    shared["project"]["seed"] = 999  # mutating in place is not
    assert other.config.get("project.seed") == 999


# ----------------------------------------------------------------------
# _style_factor_exposures
# ----------------------------------------------------------------------
def test_each_style_factor_becomes_a_cross_sectional_series() -> None:
    processed = {"value_composite": _panel(1), "mom_12_1": _panel(2), "volatility_60d": _panel(3)}
    got = _style_factor_exposures(processed, ["value", "momentum", "volatility"], SYMBOLS)
    assert set(got) == {"value", "momentum", "volatility"}
    for name, series in got.items():
        assert isinstance(series, pd.Series), f"{name} must be cross-sectional, not a panel"
        assert list(series.index) == list(SYMBOLS)


def test_the_exposure_is_the_time_mean_of_the_panel() -> None:
    panel = _panel(1)
    got = _style_factor_exposures({"value_composite": panel}, ["value"], SYMBOLS)
    np.testing.assert_allclose(got["value"].to_numpy(), panel.mean(axis=0).to_numpy())


def test_the_alias_list_is_tried_in_order() -> None:
    """The preferred alias wins even when a later one is also present."""
    preferred = _panel(1)
    fallback = _panel(2)
    got = _style_factor_exposures(
        {"value_composite": preferred, "book_to_price": fallback}, ["value"], SYMBOLS
    )
    np.testing.assert_allclose(got["value"].to_numpy(), preferred.mean(axis=0).to_numpy())


def test_a_later_alias_is_used_when_the_preferred_one_is_absent() -> None:
    fallback = _panel(2)
    got = _style_factor_exposures({"book_to_price": fallback}, ["value"], SYMBOLS)
    np.testing.assert_allclose(got["value"].to_numpy(), fallback.mean(axis=0).to_numpy())


def test_size_is_left_to_the_risk_model() -> None:
    """Market cap is handled inside the risk model, so it must not be mapped here."""
    processed = {"log_market_cap": _panel(4)}
    assert "size" not in _style_factor_exposures(processed, ["size"], SYMBOLS)


def test_a_style_factor_with_no_candidate_is_omitted() -> None:
    assert _style_factor_exposures({"mom_12_1": _panel(1)}, ["quality"], SYMBOLS) == {}


def test_a_pre_computed_series_is_used_directly() -> None:
    series = _panel(1).mean(axis=0)
    got = _style_factor_exposures({"mom_12_1": series}, ["momentum"], SYMBOLS)
    np.testing.assert_allclose(got["momentum"].to_numpy(), series.reindex(SYMBOLS).to_numpy())


def test_the_exposure_is_reindexed_onto_the_requested_symbols() -> None:
    """The risk model's asset order is the authority, not the factor panel's."""
    panel = _panel(1)
    subset = SYMBOLS[:3]
    got = _style_factor_exposures({"value_composite": panel}, ["value"], subset)
    assert list(got["value"].index) == list(subset)
    np.testing.assert_allclose(
        got["value"].to_numpy(), panel.mean(axis=0).reindex(subset).to_numpy()
    )


def test_an_empty_library_yields_no_exposures() -> None:
    assert _style_factor_exposures({}, ["value", "momentum"], SYMBOLS) == {}


# ----------------------------------------------------------------------
# _decomp_table
# ----------------------------------------------------------------------
def test_no_risk_result_means_no_table() -> None:
    assert _decomp_table(ResearchState()) is None


def test_without_weights_the_factor_covariance_is_returned() -> None:
    risk = _risk_result()
    got = _decomp_table(ResearchState(risk_result=risk, weights=None))
    assert got is risk.factor_cov


def test_the_decomposition_carries_the_covariance_for_the_chart() -> None:
    risk = _risk_result()
    weights = pd.DataFrame(0.1, index=DATES, columns=SYMBOLS)
    got = _decomp_table(ResearchState(risk_result=risk, weights=weights))
    assert got is not None
    assert got.attrs["covariance"] is risk.covariance
    assert not got.empty


def test_the_decomposition_uses_the_mean_absolute_weight() -> None:
    """A signed mean would cancel a long/short book down to nothing."""
    risk = _risk_result()
    weights = pd.DataFrame(0.0, index=DATES, columns=SYMBOLS)
    weights.iloc[:, 0] = 0.2
    weights.iloc[:, 1] = -0.2
    got = _decomp_table(ResearchState(risk_result=risk, weights=weights))
    assert got.attrs["covariance"] is risk.covariance
    # The two-sided book has real gross exposure, so the table must not be all
    # zero. The specific-risk row has no factor exposure, hence the NaN-safe sum.
    numeric = got.select_dtypes("number")
    assert np.nansum(np.abs(numeric.to_numpy())) > 0


def test_a_failed_decomposition_falls_back_to_the_factor_covariance(monkeypatch) -> None:
    risk = _risk_result()
    weights = pd.DataFrame(0.1, index=DATES, columns=SYMBOLS)

    def boom(*args, **kwargs):
        raise ValueError("incompatible shapes")

    monkeypatch.setattr("alphaforge.risk.factor_model.factor_risk_decomposition", boom)
    got = _decomp_table(ResearchState(risk_result=risk, weights=weights))
    assert got is risk.factor_cov


# ----------------------------------------------------------------------
# _notes (a staticmethod on the pipeline)
# ----------------------------------------------------------------------
def test_no_notes_for_a_bare_state() -> None:
    assert ResearchPipeline._notes(ResearchState()) == []


def test_the_survivorship_disclaimer_is_surfaced() -> None:
    state = ResearchState(config={"data": {"survivorship_bias_disclaimer": True}})
    notes = ResearchPipeline._notes(state)
    assert len(notes) == 1
    assert "Survivorship bias" in notes[0]


def test_a_false_disclaimer_is_not_surfaced() -> None:
    state = ResearchState(config={"data": {"survivorship_bias_disclaimer": False}})
    assert ResearchPipeline._notes(state) == []


def test_the_backtest_note_reports_its_diagnostics() -> None:
    backtest = SimpleNamespace(
        diagnostics={"n_rebalances": 60, "n_trades": 1200, "avg_turnover": 0.25}
    )
    notes = ResearchPipeline._notes(ResearchState(backtest=backtest))
    assert len(notes) == 1
    assert "60 rebalances" in notes[0]
    assert "1200 trades" in notes[0]
    assert "0.25" in notes[0]


def test_both_notes_appear_together() -> None:
    backtest = SimpleNamespace(diagnostics={"n_rebalances": 1, "n_trades": 2, "avg_turnover": 0.5})
    state = ResearchState(
        config={"data": {"survivorship_bias_disclaimer": True}}, backtest=backtest
    )
    assert len(ResearchPipeline._notes(state)) == 2


# ----------------------------------------------------------------------
# run_research
# ----------------------------------------------------------------------
def test_run_research_delegates_to_the_pipeline(monkeypatch) -> None:
    seen: dict = {}

    def fake_run(self, **kwargs):
        seen["config"] = self.config
        seen["kwargs"] = kwargs
        return ResearchState(config={"ok": True})

    monkeypatch.setattr(ResearchPipeline, "run", fake_run)
    got = run_research({"project": {"seed": 3}}, start="2020-01-01")
    assert isinstance(got, ResearchState)
    assert seen["kwargs"] == {"start": "2020-01-01"}
    assert seen["config"].get("project.seed") == 3


def test_run_research_without_a_config_still_builds_a_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(ResearchPipeline, "run", lambda self, **kwargs: ResearchState())
    assert isinstance(run_research(), ResearchState)


@pytest.mark.parametrize("blank", [None, {}])
def test_a_blank_config_is_accepted(blank) -> None:
    assert isinstance(ResearchPipeline(blank).config, Config)

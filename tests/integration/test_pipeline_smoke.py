"""Integration smoke test for the full research pipeline.

Runs the entire stack once on the bundled sample data and asserts that every
stage produced a real output and that the self-contained HTML report was
written. Marked ``slow`` so CI can run the fast unit suite separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alphaforge.pipeline as pipeline_module
from alphaforge.pipeline import ResearchPipeline
from alphaforge.utils.config import Config, set_global_seed


@pytest.mark.slow
def test_full_pipeline_runs_and_reports(tmp_path, monkeypatch):
    set_global_seed(42)
    cfg = Config.load(
        overrides={
            "model": {"type": "elasticnet"},
            "portfolio": {"method": "mean_variance", "assumed_ic": 0.031},
            "cost": {"commission_bps": 37.0},
        }
    )
    actual_engine = pipeline_module.BacktestEngine
    captured = {}

    def capture_engine(*args, **kwargs):
        engine = actual_engine(*args, **kwargs)
        captured["engine"] = engine
        return engine

    monkeypatch.setattr(pipeline_module, "BacktestEngine", capture_engine)
    report_dir = tmp_path / "reports"
    state = ResearchPipeline(cfg).run(
        start="2016-01-01",
        end="2024-12-31",
        model_type="ridge",
        report_dir=str(report_dir),
    )

    # Core stages must have produced real outputs.
    assert state.factor_summary is not None and len(state.factor_summary) > 0
    assert state.model_eval is not None
    assert state.risk_result is not None, "risk model must run"
    assert state.backtest is not None
    assert state.brinson is not None, "brinson attribution must run"
    assert state.factor_attr is not None, "factor attribution must run"
    assert state.report_path is not None
    assert state.panel.dates[0] == pd.Timestamp("2016-01-01")
    assert state.panel.dates[-1] == pd.Timestamp("2024-12-31")
    assert state.config["data"]["start_date"] == "2016-01-01"
    assert cfg.get("data.start_date") == "2015-01-01"
    assert state.diagnostics["assumed_ic"] == 0.031
    assert state.config["model"]["type"] == "ridge"
    assert state.model_eval.model_name == "ridge"
    assert cfg.get("model.type") == "elasticnet"
    assert state.model_eval.summary["rank_ic_mean"] != 0.031
    assert captured["engine"].ic == 0.031
    assert captured["engine"].cost_model.config.commission_bps == 37.0
    assert state.backtest.costs.sum() > 0

    report = tmp_path / "reports" / "research_report.html"
    assert report.exists() and report.stat().st_size > 1000

    m = state.backtest.metrics
    assert np.isfinite(m["cagr"])
    assert np.isfinite(m["sharpe"])
    # The risk-model Euler identity holds on the realised covariance too.
    from alphaforge.risk.factor_model import component_risk_contribution, portfolio_risk

    w = state.weights.abs().mean(axis=0)
    cov = state.risk_result.covariance.reindex(index=w.index, columns=w.index).fillna(0.0)
    crc = component_risk_contribution(w, cov)
    assert abs(crc.sum() - portfolio_risk(w, cov)) < 1e-5


# ---------------------------------------------------------------------------
# Degradation contract
# ---------------------------------------------------------------------------
# ``run`` wraps seven optional stages in ``try/except`` + a warning, so a broken
# risk model, regime classifier, stress run, attribution or report renderer
# degrades the output instead of aborting the run. That contract is invisible to
# the happy-path smoke test above: it only shows up when something fails, and a
# silent change from "warn and continue" to "raise" would not be caught by any
# test that only ever runs the pipeline on good data.
#
# Each case breaks exactly one collaborator, by name, in the pipeline's own
# namespace - the names it imported, not where they are defined.
_DEGRADATIONS = [
    (
        "FundamentalRiskModel",
        "fit",
        "risk_result",
        "Risk model skipped",
        None,  # risk_result stays None
    ),
    ("classify_regime", None, "regime", "Market regime analysis skipped", None),
    ("run_scenarios", None, "stress", "Stress testing skipped", None),
    ("brinson_attribution", None, "brinson", "Brinson attribution skipped", None),
    ("factor_attribution", None, "factor_attr", "Factor attribution skipped", None),
    ("write_report", None, "report_path", "Report generation skipped", None),
    ("ResearchCopilot", "analyze", "briefing", "Copilot briefing skipped", None),
]


def _break(monkeypatch, name: str, method: str | None) -> None:
    """Make one pipeline collaborator raise, wherever the pipeline looks it up."""
    import alphaforge.pipeline as pipeline_module

    def boom(*args, **kwargs):
        raise RuntimeError(f"{name} is deliberately broken for this test")

    if name == "ResearchCopilot":
        # Imported inside run(), so it is not a module attribute - patch the source.
        import alphaforge.agents.copilot as copilot_module

        monkeypatch.setattr(copilot_module.ResearchCopilot, "analyze", boom)
        return
    if method is None:
        monkeypatch.setattr(pipeline_module, name, boom)
    else:
        real = getattr(pipeline_module, name)

        class Broken:
            def __getattr__(self, attr):
                if attr == method:
                    return boom
                return getattr(real, attr)

        monkeypatch.setattr(pipeline_module, name, Broken)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "method", "field", "warning", "_unused"),
    _DEGRADATIONS,
    ids=[f"broken-{d[0]}" for d in _DEGRADATIONS],
)
def test_a_broken_optional_stage_degrades_instead_of_aborting(
    tmp_path, monkeypatch, name, method, field, warning, _unused
):
    import alphaforge.pipeline as pipeline_module

    warnings: list[str] = []
    monkeypatch.setattr(pipeline_module.log, "warning", lambda msg: warnings.append(str(msg)))
    _break(monkeypatch, name, method)

    set_global_seed(42)
    cfg = Config.load(overrides={"portfolio": {"method": "mean_variance"}})
    report_dir = tmp_path / "reports"
    state = ResearchPipeline(cfg).run(
        start="2019-01-01",
        end="2023-12-31",
        model_type="ridge",
        report_dir=str(report_dir),
    )

    # The run completed and said why the stage was dropped.
    assert any(warning in w for w in warnings), f"expected {warning!r} in {warnings}"
    assert getattr(state, field) is None, f"{field} should have stayed unset"
    # Everything upstream still produced real output.
    assert state.factor_summary is not None and len(state.factor_summary) > 0
    assert state.backtest is not None, "a broken optional stage must not cost the backtest"
    assert state.weights is not None


@pytest.mark.slow
def test_a_broken_risk_model_also_drops_the_stages_that_depend_on_it(tmp_path, monkeypatch):
    """The cascade is part of the contract: no risk model means no stress run."""
    import alphaforge.pipeline as pipeline_module

    warnings: list[str] = []
    monkeypatch.setattr(pipeline_module.log, "warning", lambda msg: warnings.append(str(msg)))
    _break(monkeypatch, "FundamentalRiskModel", "fit")

    set_global_seed(42)
    cfg = Config.load(overrides={"portfolio": {"method": "mean_variance"}})
    state = ResearchPipeline(cfg).run(
        start="2019-01-01",
        end="2023-12-31",
        model_type="ridge",
        report_dir=str(tmp_path / "reports"),
    )

    assert state.risk_result is None
    assert state.stress is None, "stress needs the risk model and must be skipped with it"
    # The report is still written - it just has no risk section.
    assert state.report_path is not None and state.report_path.exists()


@pytest.mark.slow
def test_a_broken_report_renderer_still_leaves_a_usable_state(tmp_path, monkeypatch):
    """The run's value is the state, not only the HTML."""

    _break(monkeypatch, "write_report", None)
    set_global_seed(42)
    cfg = Config.load(overrides={"portfolio": {"method": "mean_variance"}})
    state = ResearchPipeline(cfg).run(
        start="2019-01-01",
        end="2023-12-31",
        model_type="ridge",
        report_dir=str(tmp_path / "reports"),
    )
    assert state.report_path is None
    assert state.backtest is not None
    assert state.risk_result is not None, "a report failure must not undo the analysis"
    assert not (tmp_path / "reports" / "research_report.html").exists()


@pytest.mark.slow
def test_regime_labels_are_carried_forward_onto_rebalance_dates(tmp_path, monkeypatch):
    """A rebalance date absent from the daily benchmark index uses the last known regime.

    ``run`` reindexes the daily regime labels onto the (monthly) backtest return
    dates and fills any exact-match miss with ``asof``. With the bundled sample
    data every rebalance date is also a benchmark date, so that fallback never
    runs unless a label is deliberately removed - which is what this does.
    """
    import alphaforge.pipeline as pipeline_module

    real = pipeline_module.classify_regime

    def gapped(returns):
        regime = real(returns)
        return regime.drop(regime.index[len(regime) // 2])

    monkeypatch.setattr(pipeline_module, "classify_regime", gapped)
    set_global_seed(42)
    cfg = Config.load(overrides={"portfolio": {"method": "mean_variance"}})
    state = ResearchPipeline(cfg).run(
        start="2019-01-01",
        end="2023-12-31",
        model_type="ridge",
        report_dir=str(tmp_path / "reports"),
    )

    assert state.regime is not None
    stats = state.diagnostics.get("regime_stats")
    assert stats, "the per-regime table must survive a missing daily label"
    assert not state.regime.isna().all()

"""Integration smoke test for the full research pipeline.

Runs the entire stack once on the bundled sample data and asserts that every
stage produced a real output and that the self-contained HTML report was
written. Marked ``slow`` so CI can run the fast unit suite separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphaforge.pipeline import ResearchPipeline
from alphaforge.utils.config import Config, set_global_seed


@pytest.mark.slow
def test_full_pipeline_runs_and_reports(tmp_path):
    set_global_seed(42)
    cfg = Config.load(overrides={"portfolio": {"method": "mean_variance"}})
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

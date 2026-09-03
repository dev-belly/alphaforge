"""Fast unit tests for the deterministic copilot tool layer.

These pin the agent tools without spinning up the (slow) full pipeline, so a
regression in e.g. ``risk_decomposition`` fails the fast gate instead of only
the slow integration job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.agents.tools import risk_decomposition, run_tools
from alphaforge.risk.factor_model import RiskModelResult


def _make_risk_result() -> RiskModelResult:
    assets = ["AAA", "BBB", "CCC", "DDD"]
    factors = ["size", "value", "momentum"]
    exposures = pd.DataFrame(
        np.array(
            [
                [0.8, -0.3, 0.5],
                [0.7, 0.2, -0.4],
                [-0.6, 0.5, 0.3],
                [0.1, -0.2, 0.6],
            ]
        ),
        index=assets,
        columns=factors,
    )
    cov = pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
    factor_cov = pd.DataFrame(np.eye(len(factors)), index=factors, columns=factors)
    return RiskModelResult(
        exposures=exposures,
        factor_cov=factor_cov,
        specific_var=pd.Series(0.01, index=assets),
        covariance=cov,
        factor_returns=pd.DataFrame(
            0.0, index=pd.date_range("2024-01-01", periods=3), columns=factors
        ),
        residuals=pd.DataFrame(0.0, index=assets, columns=pd.date_range("2024-01-01", periods=3)),
        r_squared=0.51,
    )


def test_risk_decomposition_accepts_series_weights() -> None:
    rr = _make_risk_result()
    w = pd.Series([0.3, 0.3, 0.2, 0.2], index=rr.exposures.index)
    res = risk_decomposition(w, rr)
    assert res.ok
    assert "top_exposures" in res.data
    assert res.data["r_squared"] == 0.51


def test_risk_decomposition_accepts_dataframe_weights() -> None:
    # Regression: the pipeline stores weights as a (assets x dates) DataFrame,
    # not a Series. The tool must normalise it to the latest rebalance column,
    # otherwise .abs().sort_values(ascending=False) raises ValueError on a
    # DataFrame (needs 'by') and the copilot's /agent/query tool=risk 404s.
    rr = _make_risk_result()
    w = pd.DataFrame(
        {
            pd.Timestamp("2024-01-01"): [0.3, 0.3, 0.2, 0.2],
            pd.Timestamp("2024-02-01"): [0.25, 0.25, 0.25, 0.25],
        },
        index=rr.exposures.index,
    )
    res = risk_decomposition(w, rr)
    assert res.ok, res.note
    assert "top_exposures" in res.data


def test_risk_decomposition_no_model_is_false() -> None:
    w = pd.Series([0.25, 0.25, 0.25, 0.25], index=["AAA", "BBB", "CCC", "DDD"])
    res = risk_decomposition(w, None)
    assert not res.ok


def test_run_tools_risk_with_dataframe_weights() -> None:
    # End-to-end through run_tools() with the real as_tool_state() shape.
    rr = _make_risk_result()
    w = pd.DataFrame(
        {pd.Timestamp("2024-01-01"): [0.3, 0.3, 0.2, 0.2]},
        index=rr.exposures.index,
    )
    results = run_tools({"weights": w, "risk_result": rr})
    assert "risk" in results
    assert results["risk"].ok, results["risk"].note

"""Integration test for the FastAPI research service (via TestClient).

Requires the ``api`` extra (fastapi, httpx). Skipped automatically if the
dependency is absent so the unit suite stays lean.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# Make the apps/api package importable inside the test session.
APPS_API = Path(__file__).resolve().parents[2] / "apps" / "api"
if str(APPS_API) not in sys.path:
    sys.path.insert(0, str(APPS_API))

try:
    from fastapi.testclient import TestClient  # noqa: E402

    HAVE_API = True
except Exception:  # noqa: BLE001
    HAVE_API = False

pytestmark = pytest.mark.skipif(not HAVE_API, reason="fastapi not installed (api extra)")


@pytest.fixture(scope="module")
def client():
    from alphaforge_api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_run_request_seed_and_model_reach_pipeline(client, monkeypatch):
    import alphaforge_api.main as api

    from alphaforge.pipeline import ResearchState

    seen = []

    class FakePipeline:
        def __init__(self, config):
            seen.append((config.get("project.seed"), config.get("model.type")))

        def run(self, **kwargs):
            return ResearchState()

    monkeypatch.setattr(api, "ResearchPipeline", FakePipeline)
    monkeypatch.setitem(api._STATE, "state", None)
    monkeypatch.setitem(api._STATE, "ran_at", None)

    for payload in ({}, {"seed": 7, "model": "random_forest"}):
        response = client.post("/research/run", json=payload)
        assert response.status_code == 200, response.text
    assert seen == [(42, "ridge"), (7, "random_forest")]


def test_quality_query_exposes_cached_etl_report(client, monkeypatch):
    from alphaforge_api.main import _STATE

    from alphaforge.data.quality import DataQualityReport
    from alphaforge.pipeline import ResearchState

    quality = DataQualityReport(n_rows=100, n_symbols=5, coverage=0.98, provenance="sample")
    monkeypatch.setitem(_STATE, "state", ResearchState(quality=quality))
    response = client.post("/agent/query", json={"tool": "quality"})
    assert response.status_code == 200, response.text
    assert response.json()["data"] == quality.to_dict()


@pytest.fixture
def cached_backtest(monkeypatch):
    """Exercise the real accounting engine with fixed, cheap portfolio targets."""
    import alphaforge_api.main as api

    from alphaforge.features.panel import MarketPanel
    from alphaforge.pipeline import ResearchState

    dates = pd.bdate_range("2024-01-01", periods=12)
    close = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)
    panel = MarketPanel(
        dates=dates,
        close=close,
        raw_close=close.copy(),
        returns=close.pct_change(fill_method=None),
        volume=volume,
        dollar_volume=close * volume,
        market_cap=pd.DataFrame(1e9, index=dates, columns=close.columns),
        universe=pd.DataFrame(True, index=dates, columns=close.columns),
        industry=pd.DataFrame("Other", index=dates, columns=close.columns),
    )

    class ReportingOnlyEvaluation:
        @property
        def summary(self):
            raise AssertionError("OOS evaluation must not calibrate historical rebalances")

    class FixedConstructor:
        def __init__(self, panel, portfolio, construction):
            pass

        def construct(self, *, date, scores, prev_weights, ic):
            return SimpleNamespace(weights=pd.Series({"A": 0.5, "B": 0.0}))

    state = ResearchState(
        panel=panel,
        signal_panel=pd.DataFrame(1.0, index=dates, columns=close.columns),
        model_eval=ReportingOnlyEvaluation(),
        config={
            "portfolio": {"method": "equal_weight", "assumed_ic": 0.17},
            "backtest": {
                "rebalance": "daily",
                "initial_capital": 10_000.0,
                "min_history_days": 2,
                "adv_window": 5,
            },
            "cost": {
                "commission_bps": 12.0,
                "slippage_bps": 8.0,
                "impact_coeff_bps": 4.0,
                "min_commission": 0.5,
                "borrow_bps_annual": 123.0,
            },
        },
    )
    original_engine = api.BacktestEngine
    engines = []

    def recording_engine(*args, **kwargs):
        engine = original_engine(*args, **kwargs)
        engines.append(engine)
        return engine

    monkeypatch.setitem(api._STATE, "state", state)
    monkeypatch.setattr(api, "PortfolioConstructor", FixedConstructor)
    monkeypatch.setattr(api, "BacktestEngine", recording_engine)
    return state, engines


@pytest.mark.parametrize("endpoint", ["/optimize", "/backtests"])
@pytest.mark.parametrize("assumed_ic", [None, -0.2])
def test_reruns_use_ex_ante_ic_and_cached_cost_model(client, cached_backtest, endpoint, assumed_ic):
    state, engines = cached_backtest
    if assumed_ic is None:
        state.config["portfolio"].pop("assumed_ic")
    else:
        state.config["portfolio"]["assumed_ic"] = assumed_ic
    original_config = deepcopy(state.config)
    response = client.post(endpoint, json={})
    assert response.status_code == 200, response.text
    engine = engines[-1]
    assert engine.ic == pytest.approx(0.03 if assumed_ic is None else assumed_ic)
    for key, value in state.config["cost"].items():
        assert getattr(engine.cost_model.config, key) == value
    assert engine.config.initial_capital == 10_000.0
    assert state.config == original_config


def test_cost_override_changes_real_net_return_without_mutating_the_cache(client, cached_backtest):
    state, engines = cached_backtest
    original_config = deepcopy(state.config)
    free = client.post("/backtests", json={"cost_rate": 0.0})
    charged = client.post("/backtests", json={"cost_rate": 0.01, "execution_lag_days": 2})
    assert free.status_code == charged.status_code == 200
    assert free.json()["summary"]["total_return"] == pytest.approx(0.0)
    assert charged.json()["summary"]["total_return"] < 0
    assert charged.json()["summary"]["total_return"] < free.json()["summary"]["total_return"]
    assert engines[0].cost_model.config.commission_bps == 0.0
    assert engines[1].cost_model.config.commission_bps == 100.0
    assert engines[1].config.execution_lag_days == 2
    for engine in engines:
        cost = engine.cost_model.config
        assert cost.slippage_bps == cost.impact_coeff_bps == cost.min_commission == 0.0
        assert cost.borrow_bps_annual == 123.0
    assert state.config == original_config


@pytest.mark.parametrize("value", [-0.01, "NaN", "Infinity", 1e308])
def test_invalid_cost_overrides_are_rejected_before_running(client, cached_backtest, value):
    _state, engines = cached_backtest
    response = client.post("/backtests", json={"cost_rate": value})
    assert response.status_code == 422, response.text
    assert not engines


@pytest.mark.parametrize("literal", ["1e999", "NaN", "-Infinity"])
def test_raw_json_nonfinite_costs_have_json_safe_errors(client, cached_backtest, literal):
    _state, engines = cached_backtest
    response = client.post(
        "/backtests",
        content='{"cost_rate":' + literal + "}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "cost_rate must be finite"
    assert not engines


def test_unaffordable_transaction_cost_returns_a_request_error(client, cached_backtest):
    _state, engines = cached_backtest
    response = client.post("/backtests", json={"cost_rate": 100.0})
    assert response.status_code == 422, response.text
    assert "cost" in response.json()["detail"].lower()
    assert engines  # The real accounting engine rejects the unaffordable trade.


@pytest.mark.parametrize("endpoint", ["/optimize", "/backtests"])
@pytest.mark.parametrize("value", [float("nan"), 1.1, "invalid"])
def test_invalid_configured_ic_is_rejected_before_running(client, cached_backtest, endpoint, value):
    state, engines = cached_backtest
    state.config["portfolio"]["assumed_ic"] = value
    response = client.post(endpoint, json={})
    assert response.status_code == 422, response.text
    assert "portfolio.assumed_ic" in response.json()["detail"]
    assert not engines


@pytest.mark.slow
def test_run_then_serve(client, tmp_path):
    # A test must never write to a tracked path. This used to point at
    # ``research/reports``, so every full run of the suite overwrote the
    # published sample report and left the working tree dirty - the only visible
    # symptom was a changed "Generated ..." timestamp.
    r = client.post(
        "/research/run",
        json={
            "start": "2019-01-01",
            "end": "2024-12-31",
            "report_dir": str(tmp_path / "reports"),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["backtest"] is not None
    assert body["risk"] is not None
    assert body["brinson"] is not None

    assert client.get("/backtest").status_code == 200
    assert client.get("/attribution").status_code == 200
    assert client.get("/report").status_code == 200
    assert client.get("/briefing").status_code == 200

    # /optimize rebuilds the book with a different method - it must return the
    # latest rebalance's target weights, never a 500.  Regresses the bug where
    # the endpoint called PortfolioConstructor.construct() with the wrong arity.
    r_opt = client.post("/optimize", json={"method": "risk_parity"})
    assert r_opt.status_code == 200, r_opt.text
    opt = r_opt.json()
    assert opt["method"] == "risk_parity"
    assert opt["n_assets"] > 0
    assert opt["gross_exposure"] > 0
    assert opt["top_holdings"]

    # /backtests re-runs the accounting loop with overrides.
    r_bt = client.post("/backtests", json={"cost_rate": 0.001, "rebalance": "monthly"})
    assert r_bt.status_code == 200, r_bt.text
    assert r_bt.json()["summary"] is not None

    # The copilot's deterministic tool layer must serve every stage.  Regresses
    # the bug where the `risk` tool 404'd because the pipeline stores `weights`
    # as a (assets x dates) DataFrame while the tool assumed a pd.Series.
    assert client.get("/risk").status_code == 200
    for tool in ("model", "risk", "factors", "stress", "regime", "attribution", "quality"):
        r_q = client.post("/agent/query", json={"tool": tool})
        assert r_q.status_code == 200, r_q.text
        assert r_q.json()["ok"] is True
        if tool == "quality":
            assert r_q.json()["data"]["n_rows"] > 0

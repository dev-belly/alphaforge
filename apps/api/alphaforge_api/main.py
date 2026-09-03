"""FastAPI service wrapping the AlphaForge research pipeline.

The API is a thin, well-typed skin over :mod:`alphaforge.pipeline`.  It never
invents numbers: every response is derived from the *real* state object the
pipeline returns.  Long-running research runs are cached in-process so the
report and individual stage results can be polled separately.

Run locally::

    uvicorn alphaforge_api.main:app --reload --port 8000

or, from the project root with the editable install::

    alphaforge serve-api
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from alphaforge.agents.tools import CATALOG, run_tools
from alphaforge.backtest.engine import BacktestConfig, BacktestEngine
from alphaforge.pipeline import ResearchPipeline, ResearchState
from alphaforge.portfolio.constructor import PortfolioConstructor
from alphaforge.utils.config import Config, set_global_seed
from alphaforge.utils.logging import configure_logging, get_logger

log = get_logger("api")
configure_logging(level="INFO")

app = FastAPI(
    title="AlphaForge Research API",
    version="0.1.0",
    description="Institutional quant research & portfolio engineering, served over HTTP.",
)

# In-process cache of the most recent run.  A production deployment would put
# this behind a job queue + object store; for a single-node research box this
# is enough and keeps the demo dependency-free.
_STATE: dict[str, Any] = {"state": None, "ran_at": None}


# ---------------------------------------------------------------------------
# request / response models
# ---------------------------------------------------------------------------
class ResearchRequest(BaseModel):
    start: str | None = Field(None, description="Window start date (YYYY-MM-DD).")
    end: str | None = Field(None, description="Window end date (YYYY-MM-DD).")
    model: str | None = Field(None, description="ridge | elasticnet | random_forest | lightgbm")
    provider: str | None = Field(None, description="sample | local | yahoo | akshare | tushare")
    portfolio_method: str | None = Field(
        None, description="equal_weight | mean_variance | min_variance | max_sharpe | risk_parity"
    )
    target_volatility: float | None = Field(
        None, description="Annualised vol target for vol targeting."
    )
    report_dir: str = Field(
        "research/reports", description="Directory the HTML report is written to."
    )
    seed: int = Field(42, description="Global RNG seed for reproducibility.")
    persist: bool = Field(False, description="Persist the processed dataset to disk.")


class OptimizeRequest(BaseModel):
    method: str | None = Field(
        None, description="equal_weight | mean_variance | min_variance | max_sharpe | risk_parity"
    )
    target_volatility: float | None = Field(None, description="Annualised vol target.")


class BacktestRequest(BaseModel):
    execution_lag_days: int | None = Field(
        None, description="Session lag between signal and trade."
    )
    rebalance: str | None = Field(None, description="monthly | weekly | daily")
    cost_rate: float | None = Field(None, description="Proportional transaction-cost rate.")


class AgentQueryRequest(BaseModel):
    tool: str = Field(
        ...,
        description="Tool to invoke: factors | model | backtest | diagnostics | risk | "
        "attribution | regime | stress | quality | config",
    )


# ---------------------------------------------------------------------------
# JSON hygiene: numpy / pandas / NaN must become plain JSON
# ---------------------------------------------------------------------------
def _clean(value: Any, _depth: int = 0) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, _depth + 1) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    # numpy scalars / pandas types
    try:
        import numpy as np

        if isinstance(value, np.floating):
            f = float(value)
            return None if math.isnan(f) or math.isinf(f) else f
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:  # noqa: BLE001
        pass
    if _depth < 6 and hasattr(value, "to_dict"):
        try:
            return _clean(value.to_dict(), _depth + 1)
        except Exception:  # noqa: BLE001
            pass
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return None


def _df_records(df: Any, limit: int = 500) -> list[dict]:
    if df is None or not hasattr(df, "to_dict"):
        return []
    try:
        return _clean(df.head(limit).to_dict(orient="records"))
    except Exception:  # noqa: BLE001
        return []


def _serialize_state(state: ResearchState) -> dict:
    """Turn a research run into a JSON-safe summary (heavy frames omitted)."""
    bt = state.backtest
    cfg = state.config or {}

    risk = None
    if state.risk_result is not None:
        risk = _clean(
            {
                "r_squared": state.risk_result.r_squared,
                "n_assets": int(state.risk_result.exposures.shape[0]),
                "n_factors": int(state.risk_result.exposures.shape[1]),
                "style_factors": list(state.risk_result.exposures.columns),
            }
        )

    brinson = None
    if state.brinson is not None and hasattr(state.brinson, "to_dict"):
        brinson = _clean(state.brinson.to_dict())
        brinson["by_sector"] = _df_records(getattr(state.brinson, "by_sector", None), 60)

    factor_attr = None
    if state.factor_attr is not None and hasattr(state.factor_attr, "to_dict"):
        factor_attr = _clean(state.factor_attr.to_dict())
        factor_attr["betas"] = _clean(state.factor_attr.betas.to_dict())
        factor_attr["attributed_return"] = _clean(state.factor_attr.attributed_return.to_dict())

    briefing = None
    if state.briefing is not None:
        briefing = {
            "headline": state.briefing.headline,
            "findings": state.briefing.findings,
            "warnings": state.briefing.warnings,
            "checks": state.briefing.checks,
            "text": state.briefing.to_text(),
        }

    regime = None
    if (
        state.regime is not None
        and isinstance(state.regime, pd.Series)
        and not state.regime.dropna().empty
    ):
        regime = {
            "counts": {str(k): int(v) for k, v in state.regime.value_counts(dropna=True).items()},
            "stats": _clean(state.diagnostics.get("regime_stats", {})),
            "factor_ic": _clean(state.diagnostics.get("regime_factor_ic", {})),
        }

    return _clean(
        {
            "config_keys": sorted(cfg.keys()),
            "diagnostics": state.diagnostics,
            "n_factors": len(state.factor_summary) if state.factor_summary is not None else None,
            "factor_summary": _df_records(state.factor_summary, 200),
            "model_summary": (
                _clean(state.model_eval.summary) if state.model_eval is not None else None
            ),
            "risk": risk,
            "backtest": bt.summary() if bt is not None else None,
            "backtest_metrics": bt.metrics if bt is not None else None,
            "brinson": brinson,
            "factor_attribution": factor_attr,
            "regime": regime,
            "stress": (
                {nm: _clean(r.to_dict()) for nm, r in state.stress.items()}
                if state.stress is not None
                else None
            ),
            "briefing": briefing,
            "report_path": str(state.report_path) if state.report_path else None,
        }
    )


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "alphaforge-api",
        "version": "0.1.0",
        "utc": datetime.now(timezone.utc).isoformat(),
        "has_run": _STATE["state"] is not None,
    }


@app.get("/config")
def get_config() -> dict:
    return _clean(Config.load().raw)


@app.post("/research/run", response_class=JSONResponse)
def run_research(req: ResearchRequest) -> dict:
    """Run the full pipeline and cache the result for the other GET routes."""
    set_global_seed(req.seed)
    overrides: dict[str, Any] = {}
    if req.provider:
        overrides.setdefault("data", {})["provider"] = req.provider
    if req.portfolio_method:
        overrides.setdefault("portfolio", {})["method"] = req.portfolio_method
    if req.target_volatility is not None:
        overrides.setdefault("portfolio", {})["target_volatility"] = req.target_volatility
    config = Config.load(overrides=overrides)

    state = ResearchPipeline(config).run(
        start=req.start,
        end=req.end,
        model_type=req.model,
        report_dir=req.report_dir,
        persist=req.persist,
    )
    _STATE["state"] = state
    _STATE["ran_at"] = datetime.now(timezone.utc).isoformat()
    summary = _serialize_state(state)
    summary["ran_at"] = _STATE["ran_at"]
    return summary


@app.get("/research/last")
def research_last() -> dict:
    if _STATE["state"] is None:
        raise HTTPException(
            status_code=404, detail="No research run cached yet. POST /research/run first."
        )
    summary = _serialize_state(_STATE["state"])
    summary["ran_at"] = _STATE["ran_at"]
    return summary


@app.get("/factors")
def factors() -> dict:
    if _STATE["state"] is None or _STATE["state"].factor_summary is None:
        raise HTTPException(
            status_code=404, detail="No factor summary cached. POST /research/run first."
        )
    return {
        "n_factors": len(_STATE["state"].factor_summary),
        "factors": _df_records(_STATE["state"].factor_summary, 200),
    }


@app.get("/backtest")
def backtest() -> dict:
    bt = _STATE["state"].backtest if _STATE["state"] else None
    if bt is None:
        raise HTTPException(status_code=404, detail="No backtest cached. POST /research/run first.")
    return _clean({"summary": bt.summary(), "metrics": bt.metrics, "diagnostics": bt.diagnostics})


@app.get("/attribution")
def attribution() -> dict:
    if _STATE["state"] is None:
        raise HTTPException(status_code=404, detail="No run cached. POST /research/run first.")
    out: dict[str, Any] = {}
    if _STATE["state"].brinson is not None:
        b = _STATE["state"].brinson
        out["brinson"] = _clean(b.to_dict())
        out["brinson"]["by_sector"] = _df_records(getattr(b, "by_sector", None), 60)
    if _STATE["state"].factor_attr is not None:
        fa = _STATE["state"].factor_attr
        out["factor"] = _clean(fa.to_dict())
        out["factor"]["betas"] = _clean(fa.betas.to_dict())
        out["factor"]["attributed_return"] = _clean(fa.attributed_return.to_dict())
    return out


@app.get("/risk")
def risk() -> dict:
    rr = _STATE["state"].risk_result if _STATE["state"] else None
    if rr is None:
        raise HTTPException(
            status_code=404, detail="No risk model cached. POST /research/run first."
        )
    return _clean(
        {
            "r_squared": rr.r_squared,
            "n_assets": int(rr.exposures.shape[0]),
            "n_factors": int(rr.exposures.shape[1]),
            "factors": list(rr.exposures.columns),
        }
    )


@app.get("/briefing")
def briefing() -> dict:
    b = _STATE["state"].briefing if _STATE["state"] else None
    if b is None:
        raise HTTPException(status_code=404, detail="No briefing cached. POST /research/run first.")
    return {
        "headline": b.headline,
        "findings": b.findings,
        "warnings": b.warnings,
        "checks": b.checks,
        "text": b.to_text(),
    }


@app.get("/regime")
def regime() -> dict:
    r = _STATE["state"].regime if _STATE["state"] else None
    if r is None or not (isinstance(r, pd.Series) and not r.dropna().empty):
        raise HTTPException(status_code=404, detail="No regime cached. POST /research/run first.")
    return _clean(
        {
            "counts": {str(k): int(v) for k, v in r.value_counts(dropna=True).items()},
            "stats": _STATE["state"].diagnostics.get("regime_stats", {}),
            "factor_ic": _STATE["state"].diagnostics.get("regime_factor_ic", {}),
        }
    )


@app.get("/stress")
def stress() -> dict:
    s = _STATE["state"].stress if _STATE["state"] else None
    if not s:
        raise HTTPException(
            status_code=404, detail="No stress book cached. POST /research/run first."
        )
    return _clean({nm: r.to_dict() for nm, r in s.items()})


# ---------------------------------------------------------------------------
# portfolio + factor endpoints
# ---------------------------------------------------------------------------
def _require_state() -> ResearchState:
    if _STATE["state"] is None:
        raise HTTPException(
            status_code=404, detail="No research run cached yet. POST /research/run first."
        )
    return _STATE["state"]


@app.get("/portfolio/summary")
def portfolio_summary() -> dict:
    st = _require_state()
    w = st.weights
    if w is None:
        raise HTTPException(status_code=404, detail="No portfolio weights cached.")
    latest = w.iloc[:, -1]
    gross = float(latest.abs().sum())
    net = float(latest.sum())
    top = latest.reindex(latest.abs().sort_values(ascending=False).index).head(10)
    return _clean(
        {
            "n_assets": int((latest.abs() > 0).sum()),
            "gross_exposure": gross,
            "net_exposure": net,
            "cash_buffer": 1.0 - gross,
            "avg_abs_weight": float(latest.abs().mean()),
            "top_holdings": {str(k): float(v) for k, v in top.items()},
            "avg_turnover": st.diagnostics.get("avg_turnover"),
        }
    )


@app.get("/portfolio/positions")
def portfolio_positions() -> dict:
    st = _require_state()
    w = st.weights
    if w is None:
        raise HTTPException(status_code=404, detail="No portfolio weights cached.")
    latest = w.iloc[:, -1]
    latest = latest[latest.abs() > 0]
    return _clean(
        {
            "as_of": str(w.columns[-1].date())
            if hasattr(w.columns, "date")
            else str(w.columns[-1]),
            "positions": [{"symbol": str(s), "weight": float(v)} for s, v in latest.items()],
        }
    )


@app.get("/portfolio/performance")
def portfolio_performance() -> dict:
    st = _require_state()
    bt = st.backtest
    if bt is None:
        raise HTTPException(status_code=404, detail="No backtest cached.")
    return _clean({"summary": bt.summary(), "metrics": bt.metrics})


@app.get("/portfolio/risk")
def portfolio_risk() -> dict:
    st = _require_state()
    rr = st.risk_result
    if rr is None:
        raise HTTPException(status_code=404, detail="No risk model cached.")
    out = {
        "r_squared": rr.r_squared,
        "n_assets": int(rr.exposures.shape[0]),
        "n_factors": int(rr.exposures.shape[1]),
        "factors": list(rr.exposures.columns),
    }
    if st.stress is not None:
        out["stress"] = {nm: r.to_dict() for nm, r in st.stress.items()}
    return _clean(out)


@app.get("/factors/{factor_name}")
def factor_by_name(factor_name: str) -> dict:
    st = _require_state()
    fs = st.factor_summary
    if fs is None:
        raise HTTPException(status_code=404, detail="No factor summary cached.")
    if "factor" not in fs.columns:
        raise HTTPException(status_code=404, detail="Factor table has no 'factor' column.")
    row = fs[fs["factor"].astype(str) == factor_name]
    if row.empty:
        avail = fs["factor"].astype(str).tolist()[:50]
        raise HTTPException(
            status_code=404, detail=f"Factor '{factor_name}' not found. Try: {avail}"
        )
    return _clean(row.iloc[0].to_dict())


# ---------------------------------------------------------------------------
# agent + re-computation endpoints (operate on the cached run)
# ---------------------------------------------------------------------------
@app.post("/agent/query")
def agent_query(req: AgentQueryRequest) -> dict:
    st = _require_state()
    if req.tool not in CATALOG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tool '{req.tool}'. Known: {list(CATALOG)}",
        )
    results = run_tools(st.as_tool_state())
    r = results.get(req.tool)
    if r is None or not r.ok:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool}' returned no data.")
    return _clean({"tool": req.tool, "ok": r.ok, "note": r.note, "data": r.data})


@app.post("/optimize")
def optimize(req: OptimizeRequest) -> dict:
    """Re-build the portfolio with new method / vol target on the cached panel."""
    st = _require_state()
    panel = st.panel
    if panel is None or st.signal_panel is None:
        raise HTTPException(status_code=404, detail="No panel/signals cached to optimise.")
    overrides = {}
    if req.method:
        overrides["method"] = req.method
    if req.target_volatility is not None:
        overrides["target_volatility"] = req.target_volatility
    cfg = st.config.get("portfolio", {})
    cfg = {**cfg, **overrides}
    cons = PortfolioConstructor(panel, cfg, st.config.get("risk", {}))
    weights = cons.construct(st.signal_panel)
    latest = weights.iloc[:, -1]
    latest = latest[latest.abs() > 0]
    return _clean(
        {
            "method": cfg.get("method"),
            "target_volatility": cfg.get("target_volatility"),
            "n_assets": int((latest.abs() > 0).sum()),
            "gross_exposure": float(latest.abs().sum()),
            "top_holdings": {
                str(k): float(v)
                for k, v in latest.abs().sort_values(ascending=False).head(10).items()
            },
        }
    )


@app.post("/backtests")
def backtests(req: BacktestRequest) -> dict:
    """Re-run the backtest on the cached panel with the supplied overrides."""
    st = _require_state()
    panel = st.panel
    if panel is None or st.signal_panel is None:
        raise HTTPException(status_code=404, detail="No panel/signals cached to backtest.")
    bcfg = dict(st.config.get("backtest", {}))
    if req.execution_lag_days is not None:
        bcfg["execution_lag_days"] = req.execution_lag_days
    if req.rebalance is not None:
        bcfg["rebalance"] = req.rebalance
    if req.cost_rate is not None:
        bcfg["cost_rate"] = req.cost_rate
    bt = BacktestEngine(
        panel,
        constructor=PortfolioConstructor(
            panel, st.config.get("portfolio", {}), st.config.get("risk", {})
        ),
        signals=st.signal_panel,
        ic=st.model_eval.summary.get("rank_ic_mean", 0.03) if st.model_eval is not None else 0.03,
        config=BacktestConfig.from_dict(bcfg),
    ).run()
    return _clean({"summary": bt.summary(), "diagnostics": bt.diagnostics})


@app.get("/report")
def report() -> FileResponse:
    if _STATE["state"] is None or not _STATE["state"].report_path:
        raise HTTPException(
            status_code=404, detail="No report generated. POST /research/run first."
        )
    path = Path(_STATE["state"].report_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report file missing: {path}")
    return FileResponse(path, media_type="text/html", filename="alphaforge_report.html")


@app.get("/report/path")
def report_path() -> dict:
    if _STATE["state"] is None or not _STATE["state"].report_path:
        raise HTTPException(
            status_code=404, detail="No report generated. POST /research/run first."
        )
    return {"report_path": str(_STATE["state"].report_path)}


__all__ = [
    "app",
    "run_research",
    "health",
    "get_config",
    "research_last",
    "factors",
    "backtest",
    "attribution",
    "risk",
    "briefing",
    "regime",
    "stress",
    "report",
]

"""Run the full research pipeline and export every number the case study needs.

The case study must contain only figures produced by a real run, so this script
is the single source of truth: it executes the pipeline once and writes a JSON
dump under research/. Nothing here is hand-edited.

    python scripts/export_case_study.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.pipeline import ResearchPipeline
from alphaforge.risk.factor_model import (
    component_risk_contribution,
    factor_risk_decomposition,
    portfolio_risk,
)
from alphaforge.utils.config import Config, set_global_seed

OUT = Path("research/case_study_data.json")


def _f(x: Any) -> Any:
    """JSON-safe float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 6)


def _series(s: pd.Series | None, n: int | None = None) -> dict:
    if s is None or len(s) == 0:
        return {}
    s = s.dropna()
    if n:
        s = s.head(n)
    return {str(k): _f(v) for k, v in s.items()}


def main() -> int:
    set_global_seed(42)
    cfg = Config.load()
    state = ResearchPipeline(cfg).run(
        start="2016-01-01", end="2024-12-31", model_type="ridge", report_dir="research/reports"
    )

    out: dict[str, Any] = {
        "meta": {
            "start": "2016-01-01",
            "end": "2024-12-31",
            "seed": 42,
            "provider": cfg.raw.get("data", {}).get("provider"),
            "model_type": "ridge",
            "portfolio_method": cfg.raw.get("portfolio", {}).get("method"),
            "rebalance": cfg.raw.get("backtest", {}).get("rebalance"),
        }
    }

    # ---------------- data ----------------
    panel = state.panel
    out["data"] = {
        "n_symbols": int(len(panel.symbols)),
        "n_dates": int(len(panel.dates)),
        "date_min": str(pd.Timestamp(panel.dates[0]).date()),
        "date_max": str(pd.Timestamp(panel.dates[-1]).date()),
        "n_industries": int(panel.industry.iloc[0].nunique()),
        "industries": sorted({str(x) for x in panel.industry.iloc[0].dropna().unique()}),
        "etl_symbols": state.diagnostics.get("etl_symbols"),
    }

    # ---------------- factors ----------------
    fs = state.factor_summary
    if fs is not None and len(fs):
        cols = [c for c in ("ic_mean", "rank_ic_mean", "icir", "ic_std", "t_stat") if c in fs.columns]
        ranked = fs.sort_values("rank_ic_mean", ascending=False)
        out["factors"] = {
            "n_factors": int(len(fs)),
            "top10": json.loads(ranked.head(10)[cols].to_json(orient="index")),
            "bottom5": json.loads(ranked.tail(5)[cols].to_json(orient="index")),
            "best": str(ranked.index[0]),
            "best_rank_ic": _f(ranked["rank_ic_mean"].iloc[0]),
            "worst": str(ranked.index[-1]),
            "median_abs_rank_ic": _f(ranked["rank_ic_mean"].abs().median()),
            "n_positive_ic": int((ranked["rank_ic_mean"] > 0).sum()),
        }

    # ---------------- ML ----------------
    ev = state.model_eval
    if ev is not None:
        out["model"] = {
            "summary": {k: _f(v) for k, v in dict(ev.summary).items()},
            "quantile_returns": (
                {str(k): _f(v) for k, v in ev.quantile_returns.items()}
                if ev.quantile_returns is not None
                else {}
            ),
            "ic_series_len": int(len(ev.ic_series)) if ev.ic_series is not None else 0,
        }

    # ---------------- risk ----------------
    rm = state.risk_result
    if rm is not None:
        w = state.weights.abs().mean(axis=0)
        cov = rm.covariance.reindex(index=w.index, columns=w.index).fillna(0.0)
        crc = component_risk_contribution(w, cov)
        pr = portfolio_risk(w, cov)
        decomp = factor_risk_decomposition(w, rm.exposures, rm.factor_cov, rm.specific_var)
        out["risk"] = {
            "r_squared": _f(rm.r_squared),
            "n_factors": int(rm.exposures.shape[1]),
            "factor_names": [str(c) for c in rm.exposures.columns],
            "portfolio_vol_ann": _f(pr * np.sqrt(252)),
            "euler_identity_gap": _f(abs(crc.sum() - pr)),
            "top_risk_contributors": _series(crc.sort_values(ascending=False), n=8),
            "factor_risk_share": _series(decomp.set_index(decomp.columns[0]).iloc[:, 0])
            if decomp is not None and len(decomp)
            else {},
        }

    # ---------------- backtest ----------------
    bt = state.backtest
    if bt is not None:
        m = bt.metrics
        out["backtest"] = {
            "metrics": {k: _f(v) for k, v in m.items()},
            "diagnostics": {k: _f(v) for k, v in bt.diagnostics.items()},
            "n_days": int(len(bt.returns)),
            "n_trades": int(len(bt.trades)),
            "total_costs": _f(bt.costs.sum()),
            "initial_capital": _f(bt.config.get("initial_capital")),
            "final_equity": _f(bt.equity.iloc[-1]),
            "avg_holdings": _f(bt.diagnostics.get("avg_holdings")),
        }

    # ---------------- attribution ----------------
    br = state.brinson
    if br is not None:
        out["brinson"] = {
            "allocation": _f(br.allocation),
            "selection": _f(br.selection),
            "interaction": _f(br.interaction),
            "total_active": _f(br.total_active),
            "sum_of_terms": _f(br.allocation + br.selection + br.interaction),
            "approximation_gap": _f(abs(br.allocation + br.selection + br.interaction - br.total_active)),
            "by_sector": json.loads(br.by_sector.head(15).to_json(orient="records"))
            if br.by_sector is not None and len(br.by_sector)
            else [],
        }

    fa = state.factor_attr
    if fa is not None:
        out["factor_attribution"] = {
            "r_squared": _f(fa.r_squared),
            "n_observations": int(fa.n_observations),
            "residual_mean": _f(fa.residual_mean),
            "betas": _series(fa.betas),
            "attributed_return": _series(fa.attributed_return),
            "t_stats": _series(fa.t_stats),
        }

    # ---------------- regime ----------------
    out["regime"] = {
        "label_counts": state.diagnostics.get("regime_label_counts", {}),
        "stats": state.diagnostics.get("regime_stats") or {},
    }

    # ---------------- stress ----------------
    if state.stress:
        out["stress"] = {
            name: {
                "pnl_pct": _f(r.pnl_pct),
                "shock": {k: _f(v) for k, v in r.shock.items()},
                "factor_exposure": {k: _f(v) for k, v in r.factor_exposure.items()},
            }
            for name, r in state.stress.items()
        }

    out["report_path"] = str(state.report_path)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWROTE {OUT} ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

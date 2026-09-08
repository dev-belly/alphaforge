"""Run the full AlphaForge stack on the **real** S&P 500 dataset.

This is the script that produces the numbers in ``research/case_study.md``.
Nothing here is hard-coded: every figure in the case study is read back from
the returned :class:`~alphaforge.pipeline.ResearchState` and dumped to
``research/case_study_data.json``, so the narrative and the engine can never
drift apart.

Prerequisites::

    python scripts/fetch_real_data.py

Then::

    python scripts/run_real_research.py
    python scripts/run_real_research.py --model lightgbm --method max_sharpe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from alphaforge.pipeline import ResearchPipeline  # noqa: E402
from alphaforge.utils.config import Config, set_global_seed  # noqa: E402
from alphaforge.utils.logging import configure_logging, get_logger  # noqa: E402

configure_logging()
log = get_logger("run_real_research")

CONFIG_PATH = REPO_ROOT / "configs" / "real_sp500.yaml"
DATA_DIR = REPO_ROOT / "data" / "processed"
OUT_JSON = REPO_ROOT / "research" / "case_study_data.json"


def _require_dataset() -> None:
    if not (DATA_DIR / "prices.parquet").exists():
        raise SystemExit(
            "Real dataset not found. Build it first:\n"
            "    python scripts/fetch_real_data.py\n"
            "(or run the synthetic demo instead: python -m alphaforge.cli "
            "--start 2016-01-01 --end 2024-12-31)"
        )


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{float(x):+.2%}"


def _fmt_num(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{float(x):.{nd}f}"


def collect(state, cfg: dict) -> dict:
    """Flatten everything the case study needs into one JSON-serialisable dict."""
    bt = state.backtest
    d = state.diagnostics

    summary = bt.summary() if bt is not None else {}
    panel = state.panel

    payload: dict = {
        "meta": {
            "provider": d.get("data_provider"),
            "universe": cfg.get("data", {}).get("universe"),
            "start": str(panel.dates.min().date()) if panel is not None else None,
            "end": str(panel.dates.max().date()) if panel is not None else None,
            "n_symbols": int(len(panel.symbols)) if panel is not None else 0,
            "n_dates": int(len(panel.dates)) if panel is not None else 0,
            "n_factors": int(d.get("n_factors", 0)),
            "model": cfg.get("model", {}).get("type"),
            "portfolio_method": cfg.get("portfolio", {}).get("method"),
            "rebalance": cfg.get("backtest", {}).get("rebalance"),
            "initial_capital": cfg.get("backtest", {}).get("initial_capital"),
            "market_cap_source": (
                panel.metadata.get("market_cap_source") if panel is not None else None
            ),
        },
        "factors": {
            "n_evaluated": int(d.get("n_factors", 0)),
            "top": (
                state.factor_summary.head(15).to_dict(orient="records")
                if state.factor_summary is not None
                else []
            ),
        },
        "model": {
            "rank_ic_mean": d.get("rank_ic_mean"),
            **(state.model_eval.summary if state.model_eval is not None else {}),
        },
        "risk": {
            "r_squared": getattr(state.risk_result, "r_squared", None),
            "n_factors": (
                int(state.risk_result.exposures.shape[1]) if state.risk_result is not None else 0
            ),
            "euler_identity_gap": getattr(state.risk_result, "euler_identity_gap", None),
        },
        "backtest": {
            "summary": {k: (None if pd.isna(v) else float(v)) for k, v in summary.items()},
            "diagnostics": {
                k: (v if not isinstance(v, (pd.Series, pd.DataFrame)) else str(v))
                for k, v in (bt.diagnostics if bt is not None else {}).items()
            },
        },
        "regime": {
            "stats": d.get("regime_stats"),
            "label_counts": d.get("regime_label_counts"),
        },
        "attribution": {
            "brinson": (
                {
                    "total_active": state.brinson.total_active,
                    "allocation": state.brinson.allocation,
                    "selection": state.brinson.selection,
                    "interaction": state.brinson.interaction,
                }
                if state.brinson is not None
                else None
            ),
            "factor_r_squared": getattr(state.factor_attr, "r_squared", None),
        },
        "stress": d.get("stress_pnl"),
        "report_path": str(state.report_path) if state.report_path else None,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--model", default=None, help="ridge | elasticnet | random_forest | lightgbm")
    ap.add_argument(
        "--method",
        default=None,
        help="equal_weight | mean_variance | min_variance | max_sharpe | risk_parity",
    )
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--report-dir", default=str(REPO_ROOT / "research" / "reports"))
    ap.add_argument("--no-json", action="store_true", help="skip writing case_study_data.json")
    args = ap.parse_args(argv)

    _require_dataset()

    overrides: dict = {}
    if args.model:
        overrides.setdefault("model", {})["type"] = args.model
    if args.method:
        overrides.setdefault("portfolio", {})["method"] = args.method

    cfg = Config.load(args.config, overrides=overrides)
    set_global_seed(int(cfg.get("project.seed", 42) or 42))

    log.info(f"Running real-data research with provider={cfg.get('data.provider')}")
    state = ResearchPipeline(cfg).run(
        start=args.start,
        end=args.end,
        model_type=args.model,
        report_dir=args.report_dir,
    )

    payload = collect(state, cfg.raw)

    if not args.no_json:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
        log.info(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")

    # -- console summary -------------------------------------------------
    bt = state.backtest
    s = bt.summary() if bt is not None else {}
    print("\n" + "=" * 72)
    print("ALPHAFORGE · REAL-DATA RESEARCH RUN")
    print("=" * 72)
    m = payload["meta"]
    print(
        f"Universe      : {m['n_symbols']} symbols | {m['n_dates']} trading days | "
        f"{m['start']} -> {m['end']}"
    )
    print(
        f"Factors       : {m['n_factors']} evaluated | model={m['model']} | method={m['portfolio_method']}"
    )
    print(f"Size source   : {m['market_cap_source']}")
    print("-" * 72)
    print(f"Rank IC (OOS) : {_fmt_num(payload['model'].get('rank_ic_mean'))}")
    print(f"ICIR          : {_fmt_num(payload['model'].get('icir'))}")
    print(f"Risk model R² : {_fmt_num(payload['risk'].get('r_squared'))}")
    print("-" * 72)
    print(f"CAGR (net)    : {_fmt_pct(s.get('cagr'))}")
    print(f"Sharpe (net)  : {_fmt_num(s.get('sharpe'), 2)}")
    print(f"Max drawdown  : {_fmt_pct(s.get('max_drawdown'))}")
    print(f"Ann. vol      : {_fmt_pct(s.get('ann_vol'))}")
    print(f"Beta / Alpha  : {_fmt_num(s.get('beta'), 2)} / {_fmt_pct(s.get('alpha'))}")
    print("-" * 72)
    diag = bt.diagnostics if bt is not None else {}
    print(
        f"Rebalances    : {diag.get('n_rebalances')} | trades {diag.get('n_trades')} | "
        f"avg turnover {_fmt_num(diag.get('avg_turnover'), 2)}"
    )
    print(f"Total cost    : {diag.get('total_cost')}")
    print("=" * 72)
    if state.report_path:
        print(f"HTML report   : {state.report_path}")
    print("Case study in : research/case_study.md (regenerate narrative from the JSON above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

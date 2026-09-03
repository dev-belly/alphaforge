"""Diagnose extreme daily returns / NAV overflow in the backtest.

Run:  python scripts/diagnose_nav.py
"""

from __future__ import annotations

import sys
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("error", RuntimeWarning)

from alphaforge.pipeline import ResearchPipeline  # noqa: E402
from alphaforge.utils.config import Config, set_global_seed  # noqa: E402


def main() -> int:
    set_global_seed(42)
    cfg = Config.load(overrides={"portfolio": {"method": "mean_variance"}})
    try:
        state = ResearchPipeline(cfg).run(
            start="2016-01-01", end="2024-12-31", model_type="ridge", report_dir=None
        )
    except RuntimeWarning as exc:
        print("\n=== RuntimeWarning raised as error ===")
        traceback.print_exc()
        print(f"\nwarning text: {exc}")
        return 1

    r = state.backtest.returns
    eq = state.backtest.equity
    print("\n=== RETURN DIAGNOSTICS ===")
    print(f"n            : {len(r)}")
    print(f"min          : {r.min():.6g}")
    print(f"max          : {r.max():.6g}")
    print(f"mean         : {r.mean():.6g}")
    print(f"n |r|>0.5    : {int((r.abs() > 0.5).sum())}")
    print(f"n |r|>2.0    : {int((r.abs() > 2.0).sum())}")
    print(f"n inf        : {int(np.isinf(r).sum())}")
    print(f"n nan        : {int(r.isna().sum())}")

    print("\n=== EQUITY DIAGNOSTICS ===")
    print(f"first        : {eq.iloc[0]:.6g}")
    print(f"min          : {eq.min():.6g}")
    print(f"max          : {eq.max():.6g}")
    print(f"last         : {eq.iloc[-1]:.6g}")

    extreme = r[r.abs() > 0.5]
    if len(extreme):
        print("\n=== EXTREME DAYS (|ret| > 50%) ===")
        print(extreme.sort_values().head(20).to_string())

    # Where does the compounded curve overflow?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        curve = (1.0 + r.fillna(0.0)).cumprod()
    print("\n=== COMPOUND CURVE ===")
    print(f"max          : {curve.max():.6g}")
    print(f"n inf        : {int(np.isinf(curve).sum())}")

    # Per-day prev_nav reconstruction: flag near-zero denominators
    prev = eq.shift(1)
    print("\n=== SMALLEST PREV_NAV VALUES ===")
    print(prev.nsmallest(10).to_string())

    costs = state.backtest.costs
    print("\n=== COSTS ===")
    print(f"total        : {costs.sum():.6g}")
    print(f"max daily    : {costs.max():.6g}")

    m = state.backtest.metrics
    print("\n=== HEADLINE METRICS ===")
    for k in ("cagr", "sharpe", "max_drawdown", "ann_vol", "gross_cagr", "gross_sharpe"):
        print(f"{k:14s}: {m.get(k)}")

    _ = pd  # keep import for interactive use
    return 0


if __name__ == "__main__":
    sys.exit(main())

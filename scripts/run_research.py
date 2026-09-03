"""Convenience entry point: run a full AlphaForge research pass.

Equivalent to ``alphaforge --start ... --end ...`` but callable as a module so
it can be dropped into a scheduler / notebook::

    python scripts/run_research.py --start 2016-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from a source checkout without an editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alphaforge.pipeline import ResearchPipeline  # noqa: E402
from alphaforge.utils.config import Config, set_global_seed  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Run a full AlphaForge research pipeline.")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--model", default=None)
    p.add_argument("--method", default=None, help="Portfolio method override.")
    p.add_argument("--report-dir", default="research/reports")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_global_seed(args.seed)
    overrides = {}
    if args.method:
        overrides.setdefault("portfolio", {})["method"] = args.method
    cfg = Config.load(overrides=overrides)

    state = ResearchPipeline(cfg).run(
        start=args.start, end=args.end, model_type=args.model, report_dir=args.report_dir
    )
    if state.backtest is not None:
        print("Headline:", state.backtest.summary())
    if state.report_path:
        print("Report:", state.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

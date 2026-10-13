"""Reproduce a same-dataset model/baseline comparison, with source limitations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphaforge.comparison import run_comparison
from alphaforge.data.assurance import audit_local_dataset
from alphaforge.pipeline import ResearchPipeline
from alphaforge.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/real_sp500.yaml")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2019-11-14")
    parser.add_argument("--backtest-start", default="2015-01-01")
    parser.add_argument("--backtest-end", default="2019-10-31")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 42])
    parser.add_argument("--out", type=Path, default=Path("research/comparison"))
    parser.add_argument("--require-complete-sources", action="store_true")
    args = parser.parse_args()
    cfg = Config.load(
        args.config,
        overrides={
            "data": {"start_date": args.start, "end_date": args.end},
            "backtest": {"start_date": args.backtest_start, "end_date": args.backtest_end},
        },
    )
    assurance = None
    if cfg.get("data.provider") == "local":
        assurance = audit_local_dataset(
            "data/processed",
            args.start,
            args.end,
            require_complete_sources=args.require_complete_sources,
        )
        cfg.raw["data"]["assurance"] = assurance
    state = ResearchPipeline(cfg).run(report_dir=str(args.out / "base-report"))
    result = run_comparison(state, args.out, seeds=tuple(args.seeds), assurance=assurance)
    print(json.dumps(result["aggregates"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

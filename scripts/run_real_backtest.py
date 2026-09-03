"""Run AlphaForge end-to-end on *real* market data (no synthetic sample).

Usage
-----
    python scripts/run_real_backtest.py \
        --provider eastmoney --start 2018-01-01 --end 2024-12-31

This exercises the full pipeline (data -> panel -> factors -> ML walk-forward ->
risk model -> portfolio -> backtest -> attribution -> report) against a live
vendor. The default provider is ``eastmoney`` (key-less A-share data, the same
backend AkShare uses). ``yahoo`` is also supported where the network is not
rate-limited.

Because the ``eastmoney`` backend supplies price data only (no fundamentals and
no market cap), the run:
  * uses the CSI 300 (``000300``) as the benchmark, and
  * restricts the risk-model style factors to the price-based ones
    (momentum / volatility / liquidity) so the risk decomposition stays valid.

These choices are made automatically by this script; the core platform code is
unchanged.
"""

from __future__ import annotations

import argparse

from alphaforge.pipeline import ResearchPipeline
from alphaforge.utils.config import Config, set_global_seed
from alphaforge.utils.logging import configure_logging, get_logger

log = get_logger("scripts.real_backtest")

# EastMoney supplies price data only -> restrict risk style factors to those
# derivable from prices (size/value/quality need market cap / fundamentals).
PRICE_BASED_STYLE_FACTORS = ["momentum", "volatility", "liquidity"]


def main() -> int:
    ap = argparse.ArgumentParser(description="AlphaForge real-data backtest runner")
    ap.add_argument("--provider", default="eastmoney", help="eastmoney | yahoo")
    ap.add_argument("--universe", default="000300", help="Benchmark/index id (e.g. 000300 CSI300)")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--model", default="ridge")
    ap.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated universe; defaults to the provider's curated list.",
    )
    ap.add_argument("--report-dir", default="research/reports/real")
    ap.add_argument("--persist", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    configure_logging(level="INFO")
    set_global_seed(args.seed)

    overrides = {
        "data": {"provider": args.provider, "universe": args.universe},
        # Price-only vendor -> keep the risk model on price-derived style factors.
        "risk": {"style_factors": PRICE_BASED_STYLE_FACTORS},
    }
    if args.provider == "eastmoney":
        # Make the report title explicit about the data source.
        overrides["reporting"] = {"title": "AlphaForge Real-Data Report (EastMoney A-share)"}

    config = Config.load(overrides=overrides)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    log.info(
        f"Real backtest: provider={args.provider} universe={args.universe} "
        f"{args.start}..{args.end} model={args.model}"
    )
    state = ResearchPipeline(config).run(
        start=args.start,
        end=args.end,
        model_type=args.model,
        report_dir=args.report_dir,
        persist=args.persist,
        symbols=symbols,
    )

    if state.backtest is not None:
        s = state.backtest.summary()
        log.info("Headline: " + ", ".join(f"{k}={v}" for k, v in s.items() if v == v))
    if state.briefing is not None:
        print("\n" + state.briefing.to_text())
    if state.report_path:
        print(f"\nReport: {state.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

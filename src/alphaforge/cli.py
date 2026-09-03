"""Command-line entry point for AlphaForge.

A thin argparse wrapper around :mod:`alphaforge.pipeline`.  No third-party CLI
dependency is required so the command works in a bare environment.  The full
parameter set still comes from ``configs/default.yaml``; the flags only override
the knobs you change most often.
"""

from __future__ import annotations

import argparse
import sys

from alphaforge.utils.config import Config, set_global_seed
from alphaforge.utils.logging import configure_logging, get_logger

log = get_logger("cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alphaforge",
        description="Institutional quant research & portfolio engineering pipeline",
    )
    p.add_argument("--config", default=None, help="Path to a YAML config (overrides defaults).")
    p.add_argument("--start", default=None, help="Backtest/window start date (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="Backtest/window end date (YYYY-MM-DD).")
    p.add_argument(
        "--model", default=None, help="Model type: ridge | elasticnet | random_forest | lightgbm."
    )
    p.add_argument("--provider", default=None, help="Data provider: sample | local | yahoo | ...")
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated universe (e.g. AAPL,MSFT). Live providers fall "
        "back to a curated default universe when omitted.",
    )
    p.add_argument(
        "--report-dir", default="research/reports", help="Where to write the HTML report."
    )
    p.add_argument("--persist", action="store_true", help="Persist the processed dataset to disk.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging.")
    p.add_argument(
        "--print-briefing",
        action="store_true",
        help="Print the research copilot briefing to stdout.",
    )
    p.add_argument(
        "--serve-api",
        action="store_true",
        help="Launch the FastAPI research service (uvicorn) instead of a one-shot run.",
    )
    p.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for --serve-api (default 8000).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(level="DEBUG" if args.verbose else "INFO")

    if args.serve_api:
        return _serve_api(args.api_port)

    set_global_seed(args.seed)

    overrides = {}
    if args.provider:
        overrides.setdefault("data", {})["provider"] = args.provider
    if args.model:
        overrides.setdefault("model", {})["type"] = args.model

    config = (
        Config.load(args.config, overrides) if args.config else Config.load(overrides=overrides)
    )

    from alphaforge.pipeline import ResearchPipeline

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
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
    if args.print_briefing and state.briefing is not None:
        print("\n" + state.briefing.to_text())
    if state.report_path:
        print(f"\nReport: {state.report_path}")
    return 0


def _serve_api(port: int) -> int:
    """Launch the FastAPI research service via uvicorn."""
    import os
    import sys

    # Make the apps/api package importable when running from a source checkout.
    apps_api = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "apps", "api"
    )
    if apps_api not in sys.path:
        sys.path.insert(0, apps_api)
    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        log.error("uvicorn not installed - run `pip install -e '.[api]'` first.")
        return 1
    log.info(f"Starting AlphaForge research API on http://127.0.0.1:{port}")
    uvicorn.run("alphaforge_api.main:app", host="127.0.0.1", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download and normalise a **real** S&P 500 research dataset.

Why this script exists
----------------------
The bundled ``sample`` provider is synthetic: it proves the plumbing but cannot
support any claim about real markets. This script pulls three *public, key-less*
sources and writes them into AlphaForge's canonical long-format Parquet tables
so the whole platform can be run on real prices:

==================================  ==========================================
Source                              What it provides
==================================  ==========================================
HuggingFace jwigginton/...          Daily OHLCV + adjusted close + volume for
timeseries-daily-sp500              ~503 current S&P 500 members (1980-2024)
HuggingFace jwigginton/...          GICS sector / sub-industry **and the date
index-constituents-sp500            each name joined the index** (point-in-time
                                    entry -> partial survivorship control)
HuggingFace mmirmomeni/spy_daily    SPY daily closes used as the benchmark
==================================  ==========================================

Run::

    python scripts/fetch_real_data.py            # downloads into data/raw
    python scripts/fetch_real_data.py --offline  # reuse whatever is cached

Design notes
------------
* Every artefact lands in ``data/raw`` (git-ignored) or ``data/processed``, so
  the repository never carries a large binary. Re-running is cheap: each source
  is cached and only re-downloaded with ``--refresh``.
* The script is deterministic: same inputs -> byte-identical tables.
* No credentials are read or stored. All three sources are anonymous.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from alphaforge.data.providers.base import (  # noqa: E402
    CONSTITUENT_COLUMNS,
    MACRO_COLUMNS,
    PRICE_COLUMNS,
)
from alphaforge.utils.logging import configure_logging, get_logger  # noqa: E402

configure_logging()
log = get_logger("fetch_real_data")

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Source registry - anonymous, key-less, publicly mirrored.
# ---------------------------------------------------------------------------
PRICES_URL = (
    "https://huggingface.co/datasets/jwigginton/timeseries-daily-sp500/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
CONSTITUENTS_URL = (
    "https://huggingface.co/datasets/jwigginton/index-constituents-sp500/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
BENCHMARK_URL = "https://huggingface.co/datasets/mmirmomeni/spy_daily/resolve/main/train.jsonl"
# Published quarterly ratios (ROE / ROA / margins / turnover / leverage).
RATIOS_URL = (
    "https://huggingface.co/datasets/pmoe7/SP_500_Stocks_Data-ratios_news_price_10_yrs/"
    "resolve/main/sp500_daily_ratios_20yrs.zip"
)

# The research window is chosen so *every* source overlaps:
#   prices     1980-01-02 -> 2024-03-11
#   spy        2000-01-03 -> 2019-11-14
# The benchmark is the binding constraint, so the default build starts in 2000.
DEFAULT_START = "2000-01-01"
DEFAULT_END = "2019-12-31"


@dataclass
class BuildStats:
    rows: int = 0
    symbols: int = 0
    start: str = ""
    end: str = ""

    def as_dict(self) -> dict:
        return {"rows": self.rows, "symbols": self.symbols, "start": self.start, "end": self.end}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path, refresh: bool = False) -> Path:
    """Fetch ``url`` into ``dest`` unless it is already cached."""
    if dest.exists() and not refresh:
        log.info(f"cache hit: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"downloading {url}")
    try:
        resp = requests.get(url, timeout=180, stream=True)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - network failures are actionable, not fatal
        raise SystemExit(
            f"Could not download {url}\n"
            f"  reason: {exc}\n"
            "  This build needs outbound access to huggingface.co. If you are\n"
            "  offline, the bundled synthetic provider still works:\n"
            "      python -m alphaforge.cli --start 2016-01-01 --end 2024-12-31"
        ) from exc
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 22):
            if chunk:
                fh.write(chunk)
    tmp.replace(dest)
    log.info(f"saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def build_prices(raw_parquet: Path, constituents: pd.DataFrame) -> pd.DataFrame:
    """Long-format canonical price table with GICS sector attached."""
    log.info("normalising prices …")
    df = pd.read_parquet(
        raw_parquet,
        columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"],
    )
    df["date"] = pd.to_datetime(df["date"])

    sector = (
        constituents.set_index("symbol")["gics_sector"]
        .astype(str)
        .replace({"nan": "Unknown", "": "Unknown"})
    )
    df["industry"] = df["symbol"].map(sector).fillna("Unknown")

    # These are genuinely unavailable from the source. They stay NaN on purpose:
    # a fabricated market cap would silently corrupt the size factor, the size
    # neutralisation and every risk-model weight derived from them. The panel
    # layer substitutes an explicitly-labelled dollar-volume proxy instead.
    df["market_cap"] = pd.NA
    df["shares_outstanding"] = pd.NA

    df = df.dropna(subset=["adj_close"])
    df = df[df["adj_close"] > 0]
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    keep = [c for c in PRICE_COLUMNS if c in df.columns]
    return df[keep]


# Column mapping for the published-ratio source. ``scale`` converts the
# source's percentage points into the decimal fractions the factor library
# expects; turnover and leverage are already published as plain multiples.
RATIO_SOURCE_MAP = {
    "roe": ("ROE - Return On Equity", 0.01),
    "roa": ("ROA - Return On Assets", 0.01),
    "gross_margin": ("Gross Margin", 0.01),
    "net_margin": ("Net Profit Margin", 0.01),
    "asset_turnover": ("Asset Turnover", 1.0),
    "debt_to_equity": ("Debt/Equity Ratio", 1.0),
}

# The ratio CSV is truncated mid-file (it ends on ticker "NI"), so the last
# symbol's history is short. Require a near-complete history instead of
# silently carrying a half-observed name into the cross-section.
_MIN_FUNDAMENTAL_ROWS = 3000


def build_fundamentals(raw_zip: Path, symbols: list[str], lag_days: int) -> pd.DataFrame:
    """Published quarterly ratios -> canonical point-in-time fundamentals.

    The critical step is the **re-stamping of the release date**. The source
    labels every row with the fiscal quarter it *describes*, stamping that
    quarter's figures onto its **first** trading day - e.g. Q2-2005 numbers
    appear on 2005-04-01, three months before the period even closes. Used
    as-is that is textbook look-ahead bias.

    We therefore treat the quarter's *last* trading day as the period end and
    publish the figures only at ``period_end + lag_days`` (90 by default, a
    conservative stand-in for SEC filing delays). After this transform the
    existing ``FundamentalView`` point-in-time machinery behaves correctly.
    """
    import tempfile
    import zipfile

    if not raw_zip.exists():
        log.warning(f"no fundamentals archive at {raw_zip.name} - skipping")
        return pd.DataFrame({"symbol": pd.Series(dtype="object")})

    src_cols = ["Ticker", "Date", "year", "quarter"] + [c for _, (c, _) in RATIO_SOURCE_MAP.items()]
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(raw_zip) as zf:
        name = zf.namelist()[0]
        zf.extract(name, tmp)
        csv_path = Path(tmp) / name
        log.info("normalising fundamentals …")
        df = pd.read_csv(csv_path, usecols=src_cols, low_memory=False)

    df = df.rename(columns={"Ticker": "symbol"})
    df["Date"] = pd.to_datetime(df["Date"])

    counts = df.groupby("symbol").size()
    keep = counts[counts >= _MIN_FUNDAMENTAL_ROWS].index
    dropped = sorted(set(df["symbol"]) - set(keep))
    if dropped:
        log.warning(
            f"dropping {len(dropped)} symbols with an incomplete ratio history "
            f"(source file is truncated): {dropped[:8]}{' …' if len(dropped) > 8 else ''}"
        )
    df = df[df["symbol"].isin(keep)]

    # One row per (symbol, fiscal quarter): the value is constant within a
    # quarter, so collapsing keeps the frame small without losing information.
    key = ["symbol", "year", "quarter"]
    df = df.sort_values("Date")
    period_end = df.groupby(key)["Date"].transform("max")
    df = df.assign(_period_end=period_end).drop_duplicates(subset=key, keep="last")

    out = pd.DataFrame({"symbol": df["symbol"].to_numpy()})
    out["period_end"] = df["_period_end"].to_numpy()
    for target, (src, scale) in RATIO_SOURCE_MAP.items():
        out[target] = pd.to_numeric(df[src], errors="coerce").to_numpy() * scale

    # Fiscal period label + the release date the whole PIT layer keys off.
    pe = pd.to_datetime(out["period_end"])
    out["fiscal_period"] = pe.dt.to_period("Q").astype(str)
    out["report_date"] = pe + pd.Timedelta(days=int(lag_days))

    # Restrict to names the price panel actually carries.
    out = out[out["symbol"].isin(symbols)]
    out = out.sort_values(["symbol", "report_date"]).reset_index(drop=True)

    cols = ["symbol", "fiscal_period", "report_date", *RATIO_SOURCE_MAP]
    out = out[cols].dropna(subset=RATIO_SOURCE_MAP.keys(), how="all")
    log.info(
        f"fundamentals: {out['symbol'].nunique()} symbols | {len(out):,} quarterly releases | "
        f"{out['report_date'].min().date()} -> {out['report_date'].max().date()}"
    )
    return out


def build_industry(constituents: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    out = constituents.set_index("symbol")["gics_sector"].reindex(symbols)
    return (
        out.rename("industry")
        .astype(str)
        .replace({"nan": "Unknown"})
        .fillna("Unknown")
        .reset_index()
        .rename(columns={"index": "symbol"})
    )


def build_benchmark(raw_jsonl: Path) -> pd.DataFrame:
    """SPY daily closes -> canonical ``benchmark`` table.

    Stored as a **price level**, not a return series: the ETL layer differences
    it exactly once, which is what prevents the double-differencing bug that
    silently halves a benchmark's volatility.
    """
    import json

    rows = []
    with raw_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({"date": rec["Date"], "value": float(rec["Price"])})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").drop_duplicates("date")
    return df.reset_index(drop=True)


def build_constituents(constituents: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Point-in-time membership from each name's ``date_added``.

    This is the honest half of survivorship control. We know (and use) the date
    a name *entered* the index, so a 2015 IPO is not treated as investable in
    2010. We do **not** know when a name was removed, because the source only
    lists current members - so a stock that was dropped in 2012 still appears
    through its last price. That residual bias is documented, not hidden.
    """
    dates = pd.date_range(start, end, freq="ME")
    df = constituents.copy()
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    rows = []
    for d in dates:
        member = df["date_added"].notna() & (df["date_added"] <= d)
        syms = df.loc[member, "symbol"].tolist()
        if not syms:
            continue
        w = 1.0 / len(syms)
        rows.extend({"date": d, "symbol": s, "index_id": "SP500", "weight": w} for s in syms)
    out = pd.DataFrame(rows, columns=CONSTITUENT_COLUMNS)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def write_provenance(
    path: Path,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    constituents: pd.DataFrame,
    stats: BuildStats,
    fundamentals: pd.DataFrame,
    lag_days: int,
) -> None:
    n_ind = prices["industry"].nunique()
    if fundamentals.empty:
        fund_line = "| `fundamentals.parquet` | _absent_ | - |"
        fund_cov = "no fundamental coverage"
        fund_limits = [
            "2. **No fundamental data.** Value and quality factors (PE, PB, ROE, …)",
            "   require financial statements that this source does not carry. Those",
            "   factors are skipped by design - the factor registry refuses to emit a",
            "   signal it cannot compute rather than emitting a fabricated one.",
        ]
    else:
        fund_line = (
            f"| `fundamentals.parquet` | HuggingFace `pmoe7/SP_500_Stocks_Data-…-20yrs` "
            f"(quarterly ratios) | {fundamentals['symbol'].nunique()} symbols, "
            f"{len(fundamentals):,} releases, re-stamped at period-end + {lag_days}d |"
        )
        fund_cov = (
            f"{fundamentals['symbol'].nunique()} of {stats.symbols} symbols "
            f"({fundamentals['symbol'].nunique() / max(stats.symbols, 1):.0%})"
        )
        fund_limits = [
            "2. **Fundamental coverage is partial.** Quarterly ratios are available",
            f"   for {fund_cov}. Names without a statement simply carry no value",
            "   or quality signal; the factor registry skips them instead of",
            "   imputing one. Every ratio is re-stamped at fiscal period end +",
            f"   {lag_days} days before it enters the point-in-time view, because",
            "   the raw file stamps a quarter's figures on that quarter's *first*",
            "   day - using it as-is would be textbook look-ahead bias.",
        ]
    lines = [
        "# Real data provenance",
        "",
        "Generated by `scripts/fetch_real_data.py`. This file is the audit trail",
        "for the real-data research run - it records exactly where every number",
        "came from and, more importantly, what the data **cannot** support.",
        "",
        "## Sources",
        "",
        "| Table | Source | Coverage |",
        "|-------|--------|----------|",
        f"| `prices.parquet` | HuggingFace `jwigginton/timeseries-daily-sp500` "
        f"| {stats.symbols} symbols, {stats.start} -> {stats.end}, {stats.rows:,} rows |",
        f"| `industry.parquet` | HuggingFace `jwigginton/index-constituents-sp500` "
        f"(GICS sector) | {n_ind} sectors |",
        f"| `benchmark.parquet` | HuggingFace `mmirmomeni/spy_daily` (SPY close) "
        f"| {benchmark['date'].min().date()} -> {benchmark['date'].max().date()} |",
        f"| `constituents.parquet` | derived from `date_added` | "
        f"{constituents['symbol'].nunique()} symbols, monthly |",
        fund_line,
        "",
        "All sources are anonymous and key-less. No credential is read.",
        "",
        "## Known limitations",
        "",
        "Read these before quoting any result produced from this dataset.",
        "",
        "1. **Survivorship bias is reduced, not eliminated.** The source lists",
        "   *current* index members, so names that were removed from the S&P 500",
        "   during the sample never disappear from the panel. Entry dates",
        "   (`date_added`) are honoured, exit dates are unknown. Backtested",
        "   returns are therefore biased **upwards** by an unquantified amount.",
        *fund_limits,
        "3. **No market capitalisation.** Shares outstanding are unavailable. The",
        "   panel layer substitutes a trailing median dollar-volume proxy and",
        "   labels it `market_cap_source=dollar_volume_proxy` so that any",
        "   size-dependent result can be traced back to that substitution.",
        "4. **The benchmark is SPY, not the S&P 500 index**, and only its price",
        "   return is used (dividends are excluded from the source series). Alpha,",
        "   beta and information ratio are therefore measured against a",
        "   price-return proxy.",
        "5. **The sample ends in 2019**, which is where the benchmark source ends.",
        "   Later price history exists but is unusable without a benchmark.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python scripts/fetch_real_data.py --refresh   # re-download from source",
        "python scripts/run_real_research.py           # rebuild the case study",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))
    log.info(f"wrote {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument(
        "--lag-days",
        type=int,
        default=90,
        help="publication lag applied to each fiscal period's last trading day",
    )
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="never hit the network; fail if a cached copy is missing",
    )
    args = ap.parse_args(argv)

    if args.offline:
        missing = [
            p.name
            for p in (
                RAW_DIR / "sp500_daily.parquet",
                RAW_DIR / "sp500_constituents.parquet",
                RAW_DIR / "spy_daily.jsonl",
            )
            if not (RAW_DIR / p.name).exists()
        ]
        if missing:
            raise SystemExit(f"--offline requested but cached sources missing: {missing}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not args.offline:
        _download(PRICES_URL, RAW_DIR / "sp500_daily.parquet", args.refresh)
        _download(CONSTITUENTS_URL, RAW_DIR / "sp500_constituents.parquet", args.refresh)
        _download(BENCHMARK_URL, RAW_DIR / "spy_daily.jsonl", args.refresh)
        _download(RATIOS_URL, RAW_DIR / "sp500_daily_ratios.zip", args.refresh)

    constituents = pd.read_parquet(RAW_DIR / "sp500_constituents.parquet")

    prices = build_prices(RAW_DIR / "sp500_daily.parquet", constituents)
    prices = prices[
        (prices["date"] >= pd.Timestamp(args.start)) & (prices["date"] <= pd.Timestamp(args.end))
    ].reset_index(drop=True)
    if prices.empty:
        raise SystemExit("No price rows left after applying the date window - check --start/--end")

    symbols = sorted(prices["symbol"].unique())
    stats = BuildStats(
        rows=int(len(prices)),
        symbols=len(symbols),
        start=str(prices["date"].min().date()),
        end=str(prices["date"].max().date()),
    )

    industry = build_industry(constituents, symbols)
    benchmark = build_benchmark(RAW_DIR / "spy_daily.jsonl")
    members = build_constituents(constituents, args.start, args.end)
    fundamentals = build_fundamentals(RAW_DIR / "sp500_daily_ratios.zip", symbols, args.lag_days)

    # An empty macro table keeps the canonical schema complete; the ETL layer
    # tolerates an empty frame and the factor registry simply skips macro-driven
    # signals rather than inventing them.
    macro = pd.DataFrame({c: pd.Series(dtype="object") for c in MACRO_COLUMNS})

    for name, frame in (
        ("prices.parquet", prices),
        ("industry.parquet", industry),
        ("benchmark.parquet", benchmark),
        ("constituents.parquet", members),
        ("fundamentals.parquet", fundamentals),
        ("macro.parquet", macro),
    ):
        frame.to_parquet(PROCESSED_DIR / name, index=False)
        log.info(f"wrote {name} ({len(frame):,} rows)")

    write_provenance(
        PROCESSED_DIR / "DATA_PROVENANCE.md",
        prices,
        benchmark,
        members,
        stats,
        fundamentals,
        args.lag_days,
    )

    log.info(
        f"Real dataset ready: {stats.symbols} symbols | {stats.rows:,} rows | "
        f"{stats.start} -> {stats.end}"
    )
    log.info("Next: python scripts/run_real_research.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# AlphaForge

**Institutional quant research & portfolio engineering platform**

AlphaForge is an end-to-end research stack that takes you from raw market data
to a fully attributed, reproducible strategy report — without leaving Python.

```
data → panel → factors → ML (walk-forward) → risk model → portfolio → backtest → attribution → report → copilot
```

Every number in the report is produced by the same engine the CLI, the API and
the dashboard call, so they can never disagree. Nothing is hallucinated: the
research copilot reads real tool outputs and applies fixed rules.

## Features

| Layer | What it does |
|-------|--------------|
| **Data** | Pluggable providers (`sample`, `local`, `yahoo`, `akshare`). `tushare` is a **reserved** adapter slot — the wiring point exists but no live adapter ships yet (you must supply a token + implement the fetch). Point-in-time universe, survivorship flags, ETL quality gates. |
| **Factors** | 40+ cross-sectional factors across momentum, value, quality, risk, liquidity, size; winsorize / standardize / neutralize. |
| **Models** | Ridge / ElasticNet / RandomForest / LightGBM under walk-forward CV with purge + embargo; Rank-IC diagnostics. |
| **Risk** | Fundamental multi-factor model `Σ = B F Bᵀ + D`; vol targeting, Euler risk decomposition. |
| **Portfolio** | equal-weight, min-variance, mean-variance, max-sharpe (Charnes-Cooper), risk-parity; constraint-relaxation ladder. |
| **Execution** | Commission + slippage + square-root market impact; look-ahead-guarded broker simulation. |
| **Backtest** | Event-driven accounting loop; execution lag, delist handling, mark-to-market. Reports **gross vs net** Sharpe/CAGR so the cost drag is explicit. |
| **Attribution** | Brinson-Fachler (sectors) + returns-based factor attribution (styles). |
| **Market Regime** | Bull/Bear x High/Low-Vol classification from trailing info only; per-regime factor IC + portfolio stats. |
| **Stress Testing** | Scenario P&L on risk-model factor shocks (e.g. `market → -10%`, `momentum → -2σ`) + sector shocks. |
| **Report** | Self-contained HTML (base64 figures) + a deterministic copilot briefing. |
| **Apps** | FastAPI research service + Streamlit dashboard. |

## Install

```bash
git clone https://github.com/dev-belly/alphaforge.git
cd alphaforge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dashboard,viz,dev]"
```

## Quick start

Run the full pipeline from the CLI:

```bash
alphaforge --start 2016-01-01 --end 2024-12-31 --report-dir research/reports
```

Or programmatically:

```python
from alphaforge.pipeline import run_research

state = run_research(start="2016-01-01", end="2024-12-31")
print(state.backtest.summary())
print(state.report_path)          # research/reports/research_report.html
```

Serve the research API:

```bash
alphaforge serve-api            # uvicorn on :8000
# curl -X POST localhost:8000/research/run -H 'content-type: application/json' \
#      -d '{"start":"2019-01-01","end":"2024-12-31"}'
```

Launch the dashboard:

```bash
streamlit run apps/dashboard/streamlit_app.py
```

## Layout

```
src/alphaforge/      the library (data … attribution, reporting, agents, pipeline, cli)
apps/api/            FastAPI research service
apps/dashboard/      Streamlit dashboard
configs/             default.yaml (every knob lives here)
tests/               unit / integration / regression suites
scripts/             run_research.py + helpers
docs/                mkdocs documentation
research/reports/    generated HTML reports
```

## Testing

```bash
pytest -m "not slow"        # fast unit + regression (no heavy pipeline)
pytest                       # everything, including the slow pipeline/API runs
```

## Configuration

All strategy parameters live in `configs/default.yaml`. The CLI/API/SDK only
override the knobs you change most often. Nothing is hard-coded in the engine.

## Reproducibility

`alphaforge.utils.config.set_global_seed` seeds every RNG. The same config +
seed produces the same report. The copilot's findings are rule-driven, so a
reviewer can trace every sentence to a metric.

## Disclaimer

AlphaForge is a research tool. Figures are research outputs, **not** investment
advice. The bundled sample provider supplies point-in-time membership only and
therefore carries a survivorship-bias disclaimer (see `data.survivorship_bias_disclaimer`).

## Limitations

AlphaForge is an engineering-quality research harness, **not** a production
trading system or an investment product. Be explicit about what it is and is not:

* **Synthetic / sampled data is not real.** The bundled `sample` provider is
  fully synthetic but point-in-time. Any numbers it produces demonstrate the
  *pipeline*, not a tradeable edge. The survivorship-bias disclaimer in the data
  layer is there for a reason.
* **Survivorship handling is honest but not perfect.** Delisted names are
  force-liquidated only after a grace window rather than dropped, which avoids an
  *undeclared* survivorship bias — but the universe itself still reflects
  point-in-time membership and is only as good as the upstream provider.
* **Costs are a model.** Commission + slippage are linear in notional; market
  impact follows the square-root law. The multi-day work of a large order is
  charged at the capped-day impact but its *timing risk* (the market moving
  against the unfilled remainder) is **not** modelled, so very large orders are
  mildly optimistic. Gross vs net metrics are reported precisely so this drag is
  visible, not hidden.
* **Historical ≠ future.** Walk-forward CV, purge/embargo and FDR screening exist
  to fight overfitting; they do not guarantee out-of-sample performance. ICIR and
  the Benjamini-Hochberg pass rate are the honest bars, not the in-sample IC.
* **`tushare` is reserved, not live.** The provider slot exists; no live adapter
  ships. Plug in your own token + fetch before claiming live-data coverage.
* **Regime & stress are research aids, not signals.** They label the past and
  shock the current book; they do not forecast the next regime.
* **Single-node, in-process cache.** The API caches the last run in memory; a
  multi-user deployment needs a job queue + object store in front of it.

## Case Study (sample data, illustrative)

Running the full pipeline on the synthetic `sample` provider
(`alphaforge --start 2016-01-01 --end 2024-12-31`) produces, by construction:

* a **risk-model R²** around 0.5 — the style factors explain roughly half of
  cross-sectional variance (the rest is specific risk `D`);
* a **factor-attribution R²** around 0.75 — most of the portfolio's excess return
  is explained by its style exposures;
* a **Brinson active return** within ~1e-3 of the three-term allocation +
  selection + interaction split (the known approximation gap);
* **gross vs net** Sharpe/CAGR that diverge by exactly the charged cost drag,
  confirming costs are accounted for end-to-end;
* a **regime split** (Bull/Bear × High/Low-Vol) and a **stress book**
  (e.g. `market_drawdown_10pct`, `momentum_crash_2sigma`) showing the portfolio's
  factor-driven loss under named adverse paths.

These figures are *reproducible* (same config + seed → same report) and are
meant to validate the plumbing. Replace `sample` with a real provider before
reading them as market insight.

## License

MIT

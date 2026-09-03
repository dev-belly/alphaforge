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
| **Data** | Pluggable providers (`sample`, `local`, `yahoo`, `akshare`, `tushare`); point-in-time universe, survivorship flags. |
| **Factors** | 40+ cross-sectional factors across momentum, value, quality, risk, liquidity, size; winsorize / standardize / neutralize. |
| **Models** | Ridge / ElasticNet / RandomForest / LightGBM under walk-forward CV with purge + embargo; Rank-IC diagnostics. |
| **Risk** | Fundamental multi-factor model `Σ = B F Bᵀ + D`; vol targeting, Euler risk decomposition. |
| **Portfolio** | equal-weight, min-variance, mean-variance, max-sharpe (Charnes-Cooper), risk-parity; constraint-relaxation ladder. |
| **Execution** | Commission + slippage + square-root market impact; look-ahead-guarded broker simulation. |
| **Backtest** | Event-driven accounting loop; execution lag, delist handling, mark-to-market. |
| **Attribution** | Brinson-Fachler (sectors) + returns-based factor attribution (styles). |
| **Report** | Self-contained HTML (base64 figures) + a deterministic copilot briefing. |
| **Apps** | FastAPI research service + Streamlit dashboard. |

## Install

```bash
git clone https://github.com/dev-belly/AlphaForge.git
cd AlphaForge
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

## License

MIT

# Quickstart

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dashboard,viz,dev]"
```

## CLI

```bash
alphaforge --start 2016-01-01 --end 2024-12-31 --report-dir research/reports
```

Flags only override the knobs you change most often; everything else comes from
`configs/default.yaml`. Use `--print-briefing` to dump the copilot text.

## Python

```python
from alphaforge.pipeline import run_research

state = run_research(start="2016-01-01", end="2024-12-31")
print(state.backtest.summary())
print("report:", state.report_path)
```

## API

```bash
alphaforge serve-api --api-port 8000
```

```bash
curl -X POST localhost:8000/research/run \
     -H 'content-type: application/json' \
     -d '{"start":"2019-01-01","end":"2024-12-31"}'
curl localhost:8000/report        # serves the generated HTML
```

Interactive docs at `http://127.0.0.1:8000/docs`.

## Dashboard

```bash
streamlit run apps/dashboard/streamlit_app.py
```

## Tests

```bash
pytest -m "not slow"     # fast unit + regression
pytest                   # includes the slow full-pipeline + API runs
```

## Configuration

Edit `configs/default.yaml`. Notable sections:

| key | meaning |
|-----|---------|
| `data.provider` | `sample` \| `local` \| `yahoo` \| `akshare` \| `tushare` |
| `model.type` | `ridge` \| `elasticnet` \| `random_forest` \| `lightgbm` |
| `portfolio.method` | `equal_weight` \| `mean_variance` \| `min_variance` \| `max_sharpe` \| `risk_parity` |
| `portfolio.target_volatility` | annualised vol target (de-levers into cash if binding) |
| `backtest.rebalance` | `daily` \| `weekly` \| `monthly` |
| `risk.style_factors` | style factors for the fundamental risk model |

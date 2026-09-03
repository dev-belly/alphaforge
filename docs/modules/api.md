# API — `apps/api`

A thin, well-typed FastAPI skin over `alphaforge.pipeline`. It never invents
numbers: every response is derived from the *real* `ResearchState` the pipeline
returns. Long-running runs are cached in-process, so stage results can be polled
separately.

```bash
alphaforge serve-api            # uvicorn on :8000
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | liveness + `has_run` flag |
| GET | `/config` | effective configuration |
| POST | `/research/run` | run the full pipeline, cache the result |
| GET | `/research/last` | fetch the cached run summary |
| GET | `/factors` | factor summary table |
| GET | `/factors/{factor_name}` | one factor's evaluation row |
| GET | `/backtest` | backtest summary + metrics (incl. gross vs net) |
| GET | `/attribution` | Brinson + returns-based factor attribution |
| GET | `/risk` | risk-model R² + factor list |
| GET | `/regime` | regime counts + per-regime stats |
| GET | `/stress` | stress scenario P&L book |
| GET | `/briefing` | deterministic copilot briefing |
| GET | `/portfolio/summary` | exposures, turnover, cash buffer |
| GET | `/portfolio/positions` | latest holdings |
| GET | `/portfolio/performance` | backtest summary + metrics |
| GET | `/portfolio/risk` | risk + stress for the book |
| POST | `/optimize` | rebuild the portfolio with a new method / vol target |
| POST | `/agent/query` | invoke one copilot tool (`factors`, `model`, `backtest`, `risk`, `attribution`, `regime`, `stress`, `quality`, `config`, …) |
| POST | `/backtests` | re-run the backtest with execution-lag / rebalance / cost overrides |
| GET | `/report` · `/report/path` | the generated HTML report |

All numeric responses are JSON-safe (`numpy`/`NaN` → `null`); the same endpoint
contract is exercised by `tests/integration/test_api.py`.

```bash
curl -X POST localhost:8000/research/run \
     -H 'content-type: application/json' \
     -d '{"start":"2019-01-01","end":"2024-12-31"}'
```

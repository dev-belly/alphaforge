# AlphaForge

**Institutional quant research & portfolio engineering platform**

AlphaForge is an end-to-end research stack: raw market data in, a fully
attributed, reproducible strategy report out. Every number is produced by the
same engine that the CLI, the API and the dashboard call.

## What it does

```
data → panel → factors → ML (walk-forward) → risk model → portfolio → backtest → attribution → report → copilot
```

- **Data** — pluggable providers, point-in-time universe, survivorship flags.
- **Factors** — 40+ cross-sectional factors; winsorize / standardize / neutralize.
- **Models** — Ridge / ElasticNet / RandomForest / LightGBM under walk-forward CV.
- **Risk** — fundamental multi-factor model `Σ = B F Bᵀ + D`; Euler decomposition.
- **Portfolio** — five solvers with a constraint-relaxation ladder.
- **Execution** — commission + slippage + square-root impact; look-ahead guarded.
- **Backtest** — event-driven accounting; execution lag + delist handling.
- **Attribution** — Brinson-Fachler + returns-based factor attribution.
- **Report** — self-contained HTML + deterministic copilot briefing.
- **Apps** — FastAPI service + Streamlit dashboard.

Continue to [Architecture](architecture.md) or the [Quickstart](quickstart.md).

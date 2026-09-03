# Risk · Market Regime & Stress Testing

Two complementary research aids sit on top of the covariance-based risk model
(`Σ = B F Bᵀ + D`). The risk model tells you the *distribution* of outcomes;
regime and stress pin a single interpretation or adverse path.

## Market Regime — `alphaforge.risk.regime`

Each trading day is labelled along two independent axes inferred **only** from
trailing information (no future data leaks):

* **trend** — Bull / Bear: is the cumulative index above its trailing MA?
* **volatility** — High / Low: is rolling annualised vol above its own median?

The four combined labels are `Bull/LowVol`, `Bull/HighVol`, `Bear/LowVol`,
`Bear/HighVol`.

```python
from alphaforge.risk.regime import classify_regime, regime_statistics

regime = classify_regime(market_returns)          # pd.Series of str labels
stats  = regime_statistics(strategy_returns, regime)   # per-regime ann return/vol/Sharpe
```

* `classify_regime` degrades to empty (no labels, no crash) when there is not
  enough history to decide.
* `factor_performance_by_regime(ic_series, regime)` reports Rank-IC diagnostics
  per regime so you can see which factors work in which environment.

## Stress Testing — `alphaforge.risk.stress`

Answers *"what happens to the portfolio if factor X moves by a given amount?"*
Every scenario is a shock to a risk-model factor, so the result is internally
consistent with the risk decomposition:

```python
from alphaforge.risk.stress import run_scenarios, DEFAULT_SCENARIOS

book = run_scenarios(weights, risk_result)   # dict[str, StressResult]
worst = min(book.values(), key=lambda r: r.pnl_pct)
```

Default scenarios include `market_drawdown_10pct`, `momentum_crash_2sigma`,
`value_selloff_5pct`, `volatility_spike_3sigma`, `quality_rotation_5pct`,
`liquidity_dryup_2sigma`. Two shock kinds are supported:

* `factor` — a fixed move, e.g. `market → -10%`.
* `factor_sigma` — `k · σ` of the factor's standalone volatility.

`sector_shock(weights, industry, sector, shock_pct)` applies a direct,
asset-level shock to one GICS sector when no factor model is needed.

Both modules are fully deterministic and reproducible; the copilot surfaces them
through the `regime` and `stress` tools.

# Backtesting — `alphaforge.backtest`

The backtest engine is deliberately a **pure accounting loop**: it owns no alpha
logic. All portfolio decisions are delegated to a `weight_fn` (by default a
`PortfolioConstructor`). Its job is to turn target weights into a return series
*honestly* — guarding look-ahead, charging real costs, and handling delistings
without inventing survivorship.

```python
from alphaforge.backtest import BacktestEngine, BacktestConfig, run_backtest

bt = run_backtest(panel, signals=signal_panel, constructor=cons, ic=rank_ic, cfg=cfg)
metrics = bt.metrics            # net (after-cost)
gross = bt.gross_equity         # pre-cost curve (cost-drag overlay)
```

## Design contract

1. **Signals are generated on the rebalance date using only data available on
   that date.** The constructor receives the panel up to and including `t`.
2. **Orders are executed `execution_lag_days` sessions later** (default 1), at
   that session's close. This is the structural guard against look-ahead: a
   signal can never be traded on the same close that produced it.
3. **The day's return is earned on the pre-trade book**, and trading costs are
   deducted from NAV at the end of the session — so costs reduce the compounding
   base exactly as they do in production, and net Sharpe/CAGR are never inflated.
4. **Untradeable names are not silently dropped.** A name with no price on the
   execution date keeps its position (marked at the last valid close); a name
   that has been dark for `delist_grace_days` sessions is force-liquidated at the
   last valid price. Dropping them instead is how backtests acquire an undeclared
   survivorship bias.

## Configuration — `BacktestConfig`

| Field | Default | Meaning |
|-------|---------|---------|
| `rebalance` | `monthly` | rebalance cadence |
| `initial_capital` | 10,000,000 | NAV base |
| `execution_lag_days` | 1 | look-ahead guard |
| `min_history_days` | 252 | risk-model warm-up; first rebalances skipped until the window fills |
| `adv_window` | 20 | ADV for impact |
| `max_stale_days` | 5 | mark-to-market ffill limit |
| `delist_grace_days` | 5 | dark sessions before force-liquidation |
| `max_gross_leverage` | 1.0 | long-only budget |

## Costs — gross vs net, made explicit

Transaction costs are modelled by `CostModel` (commission + slippage +
square-root market impact) inside `BrokerSimulator`, and deducted per trade. The
engine then reconstructs the **pre-cost (gross)** return series by adding the
per-day cost drag back into the net series — *exact, no re-run*. Gross and net
Sharpe / CAGR / volatility / Sortino / Calmar / MaxDD are reported side by side,
and `cost_drag_cagr = gross_cagr - net_cagr` states the real transaction-cost
burden in one number. The `gross_equity` curve is shipped for the cost-drag
overlay in the report.

> The benchmark is stored as a daily *return* series (the ETL layer converts
> provider price levels exactly once). `ensure_returns` makes that contract
> explicit: a return series is never double-differenced (an earlier bug turned a
> ~15%-vol benchmark into a 58,000%-vol one and silently destroyed beta/alpha),
> while a level series handed in directly still works.

## Metrics — `performance_stats`

Every statistic is computed from the *realised* return series, so it already
includes whatever the strategy paid. Definitions are stated because
implementations differ:

* **Sharpe** — `(mean - rf/period) / std * sqrt(periods)`, excess over the
  period's risk-free accrual, not over zero.
* **Sortino** — same numerator, denominator = downside deviation vs a zero target
  with **all** observations in the denominator (Sortino-Satchell), so a flat
  series does not get an infinite ratio by accident.
* **Max drawdown** — worst peak-to-trough of the compounded curve.
* **Alpha / beta / IR / tracking error** — Jensen's alpha from an OLS of excess
  strategy returns on excess benchmark returns, annualised from the intercept.
* Annualisation is **inferred from the index**, never assumed, and reported
  alongside the numbers that depend on it.

## Output — `BacktestResult`

`equity`, `returns`, end-of-day `weights`, rebalance-date `target_weights`,
`trades`, `turnover`, `costs`, `metrics`, `gross_equity`, `benchmark`, plus a
`diagnostics` dict (`n_rebalances`, `n_skipped_rebalances`, `total_costs`,
`cost_drag_ann`, `avg_turnover`, `avg_holdings`, `avg_gross_exposure`). The
reporting layer and the API consume exactly this object — no recomputation, no
disagreement.

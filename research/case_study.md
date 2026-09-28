# Case Study — Synthetic Sample Backtest (2019–2024)

> **Honesty first.** Every number below is produced by the *real* AlphaForge
> engine on the bundled **synthetic** `sample` provider (seed 42). The sample
> generator injects realistic but deliberately *moderate* cross-sectional factor
> structure, so the economic signal is intentionally weak. The point of this
> case study is to demonstrate the **engineering** — data integrity, factor
> validation, walk-forward CV, transaction-cost-aware backtesting, attribution
> and reproducibility — not to present a tradable strategy. On live data the
> same pipeline is what you would point at a real vendor (Yahoo / AkShare).

The machine-readable source for everything here is
[`research/case_study_data.json`](case_study_data.json); the rendered HTML is
[`research/reports/research_report.html`](reports/research_report.html).

## Setup

| Dimension | Value |
|-----------|-------|
| Universe | 160 synthetic names, 11 GICS sectors |
| Window | 2015-01-01 → 2024-12-31 (2,609 trading days) |
| Factors | 42 across 7 categories |
| ML model | Ridge, walk-forward CV with purge + embargo |
| Portfolio | mean-variance, long-only, target vol 12% |
| Rebalance | monthly, execution lag 1 session |
| Capital | 10,000,000 simulated units (initial) |

## Factor research

42 factors are computed, winsorised, standardised and industry/market-cap
neutralised, then evaluated with per-date Pearson and Rank-IC. The **top factors
by ICIR** cluster around **0.21–0.26** — economically meaningful but modest, which
is the intended difficulty of the synthetic panel:

| Factor id | Rank-IC | ICIR | t-stat |
|-----------|---------|------|--------|
| 1 | 0.0490 | 0.256 | 2.77 |
| 9 | 0.0555 | 0.247 | 2.53 |
| 12 | 0.0503 | 0.228 | 2.23 |

Because 42 factors are screened at the 5% level, the summary table also reports a
Benjamini-Hochberg FDR flag so the reader can see which factors survive
multiple-testing correction rather than trusting raw p-values.

## ML alpha (walk-forward)

The model is evaluated out of sample. Its aggregate IC below is a diagnostic;
portfolio construction uses the fixed, predeclared `assumed_ic: 0.03`, so
historical rebalances cannot see later test outcomes.

| Metric | Value |
|--------|-------|
| Rank-IC (mean) | 0.0465 |
| ICIR | 0.215 |
| t-stat | 1.84 |
| Positive-IC ratio | 57.9% |
| Long-short IR | 0.126 |
| Out-of-sample periods | 1,545 |

## Risk model

A fundamental multi-factor model `Σ = B F Bᵀ + D` with **17 factors** (market,
size, value, momentum, volatility, liquidity, quality + 10 sector dummies)
explains **R² = 0.50** of cross-sectional variance — i.e. half the risk is
systematic and half is idiosyncratic, exactly the regime a real risk model
operates in. Euler risk contributions are exact (`euler_identity_gap = 0`).

## Backtest — gross vs net

Costs are modelled (commission + slippage + square-root impact) and deducted per
trade; the engine also reconstructs the pre-cost (gross) curve so the drag is
explicit.

| Metric | Net (after-cost) | Gross (pre-cost) |
|--------|------------------|------------------|
| Total return | 5.20% | 6.15% |
| CAGR | 0.82% | 0.97% |
| Annualised vol | 10.75% | 10.75% |
| Sharpe | 0.130 | 0.143 |
| Sortino | 0.184 | 0.203 |
| Max drawdown | −19.1% | −19.0% |
| Calmar | 0.043 | 0.051 |
| Cost drag (CAGR gap) | — | **0.15% / yr** |

Headline diagnostics: **60 rebalances** (1 skipped — insufficient scored names),
**1,855 trades**, **87,056 simulated units in total costs** (~0.87% of initial
capital over the full sample), average **23.7 holdings**, average gross exposure **0.65**, beta
**≈ 0.37** versus the SP500 sample benchmark (≈70% of variance explained by the
market, R²=0.70) — a modestly market-exposed long-only book, not market-neutral.

## Interpretation

* **Reproducibility.** The recorded seed, config, source hashes, Python and
  dependencies are in [`docs/sample/manifest.json`](../docs/sample/manifest.json).
  `python scripts/publish_sample.py` rebuilds the public example.
* **Cost honesty.** The gross-sharpe / net-sharpe gap (0.143 vs 0.130) is small
  because average turnover is only 0.19; the report states the drag in basis
  points, not prose.
* **Benchmark exposure.** Versus the SP500 sample benchmark the book carries
  beta ≈ 0.37 (R²=0.70) and information ratio ≈ 0.21. The active return is
  positive in this synthetic run and does not establish a live edge.
* **Look-ahead guarded.** Signals are executed one session after the signal
  date; the historical portfolio uses the predeclared IC rather than the full
  test-window IC; the benchmark return series is never double-differenced.
* **Candid weakness.** On this synthetic panel the strategy is barely
  positive — CAGR 0.82% net, Sharpe 0.13. That is the honest result of moderate
  injected signal plus real costs, and it is the correct answer to report. The
  engineering (validation, CV, costs, attribution) is what transfers to a real
  book; the alpha itself would be re-estimated on live data.

# Real-market comparison with declared source limits

[Open the complete comparison](experiments/real-sp500/index.html)

This exploratory experiment uses 1,173,649 real mirrored price rows, a frozen
840,736-row model dataset with 24 usable features, and 551,698 shared out-of-sample
forecasts. It compares equal weight, a fixed 12–1 momentum signal, Ridge,
LightGBM and Ridge without the explicit size predictor. Model runs use seeds
11, 29 and 42; every method uses the same forecast coverage, dated outcomes,
monthly rebalance, one-session execution lag and declared cost rules.

| At base assumed costs | Net CAGR | Net Sharpe | Maximum drawdown |
| --- | --- | --- | --- |
| Equal-weight baseline | 12.75% | 1.141 | −18.70% |
| Single-momentum baseline | 11.38% | 0.871 | −23.40% |
| Ridge | 12.85% | 1.005 | −22.06% |
| LightGBM, seed mean | 18.23% | 1.337 | −21.46% |
| Ridge without size predictor | 12.24% | 0.955 | −21.68% |

These are the measured results of the declared protocol, **not estimates of
tradable alpha**. In this sample Ridge has lower Sharpe than the simpler
equal-weight baseline. Three LightGBM seeds show variation; deterministic Ridge
does not. All 22 seed/cost cases, including doubled-cost scenarios, are retained.
Seed variation does not estimate uncertainty across market histories.

## Source and comparison boundaries

The source lists current S&P 500 members with entry dates, without complete
removed/delisted histories. Fundamentals use assumed publication lags, and
market capitalization is unavailable, so size uses a dollar-volume proxy.
Historical source vintages and classification changes are not independently
verified. These limits are recorded in the downloadable JSON and report banner.
`--require-complete-sources` rejects the bundled mirror's incomplete source
contract rather than treating real prices as proof of point-in-time correctness.

The forecast window is March 2014–October 2019; portfolio results cover January
2015–October 2019. The dataset ends November 14, 2019, where the benchmark source
ends. Benchmark coverage is checked for every price session. Forecast failures,
nonfinite predictions and differences in forecast coverage cannot be silently
dropped from a comparison.

All score-based portfolios equal weight their top 50 eligible names, with
symbol-based tie breaking and no optimizer. The equal-weight baseline holds all
forecastable eligible names, so its diversification differs. The size ablation
removes only `log_market_cap` from model inputs; it does not undo earlier size
neutralization. This experiment neither tunes parameters nor chooses a seed or
model for deployment based on the held-out results.

## Reproduction and verification

```bash
python scripts/fetch_real_data.py --start 2010-01-01 --end 2019-11-14
python scripts/run_comparison.py --out research/comparison
python scripts/verify_comparison.py research/comparison
```

The builder records raw/processed file hashes and source properties in
`DATA_ASSURANCE.json`. Comparison outputs include dataset identity, exact
parameters, per-fold metrics, per-run metrics, daily returns and an artifact
manifest. The verifier recomputes portfolio headlines and seed aggregates from
the saved return CSV without refitting models. Repeat with the recorded library
versions; vendor mirror updates can change source hashes.

Real runs default to `research/real_runs` and comparisons to
`research/comparison`. They are separate from the published synthetic
`research/case_study.md` and its JSON; that case retains its synthetic identity.

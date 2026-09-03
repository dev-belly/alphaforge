# AlphaForge — Final Engineering Report

> Institutional Quant Research & Portfolio Engineering Platform
> Repo: `dev-belly/alphaforge` (public) · Python 3.11 · pandas 3 · NumPy 2 · CVXPY (CLARABEL) · FastAPI · Streamlit · pytest · ruff

This report is written for two audiences: (a) an engineer reviewing the GitHub,
and (b) a candidate preparing to defend the project in a quant interview. Every
number in **§7 Performance Results** comes from a real, reproducible run on the
shipped *synthetic sample* dataset — nothing is fabricated (see §8 for the honest
limits of that data).

---

## 1. Project Summary

AlphaForge is an end-to-end quantitative research and portfolio engineering
platform. It covers the full research stack as decoupled, testable Python
modules:

```
Data → Panel → Factors → ML Alpha → Risk Model → Portfolio → Backtest → Attribution → Report → Copilot
```

The quant engine (`src/alphaforge/`) is fully decoupled from the delivery layer
(`apps/api` FastAPI, `apps/dashboard` Streamlit), so the research logic can be
used, tested, and imported without ever starting a server or a UI. The design
priority order follows the brief:

**Correctness > Research Integrity > Architecture > Reproducibility > Testing >
Engineering Quality > Visual Appearance > Number of Technologies.**

---

## 2. Architecture

```
src/alphaforge/
├── data/            providers (sample/local/yahoo/eastmoney), storage, pipeline,
│                   calendar, quality, universe
├── features/        MarketPanel (wide aligned panels), fundamentals
├── factors/         library, base/registry, momentum/value/quality/risk/
│                   liquidity, preprocessing, evaluation
├── models/          dataset, estimators (ridge/en/rf/lgbm), split (walk-forward),
│                   evaluation, pipeline
├── portfolio/       expected_returns, constructor, optimizer (CVXPY)
├── risk/            covariance (sample/ledoit-wolf/ewma), factor_model (Barra-style),
│                   regime, stress
├── backtest/        engine (event/vectorised hybrid), metrics
├── execution/       broker (slippage/market-impact), costs
├── attribution/     brinson, factor
├── reporting/       charts, report (HTML)
├── agents/          tools (tool-calling), copilot
└── utils/           config, logging, math_utils (annualisation, ensure_returns, …)
```

Data flow is strictly forward-only. The `DataPipeline` is the single entry point
(`src/alphaforge/data/pipeline.py`); every downstream stage only sees earlier,
real outputs. The `ResearchPipeline` (`src/alphaforge/pipeline.py`) orchestrates
all stages and isolates each in try/except so one failing layer still yields a
report from the rest.

---

## 3. Implemented Features (verified against source)

| Layer | What is actually implemented |
|---|---|
| **Data** | `DataProvider` interface + `SampleProvider` (deterministic synthetic, default), `LocalParquetProvider`, `YahooFinanceProvider`, `EastMoneyProvider` (key-less A-share). Incremental update, local cache, validation, missing-value handling, duplicate detection, adjusted prices, trading-calendar alignment, `DataQualityReport` (missing %/outlier %/stale %/coverage). |
| **Factors** | 40+ factors registered in a `REGISTRY`: Momentum (`mom_20/60/120/12_1/6_1`, `industry_momentum`, `residual_momentum`); Value (`pe/pb/ps`, `earnings_yield`, `book_to_price`, `sales_to_price`, `ebit_to_ev`, `fcf_yield`, `value_composite`); Quality (`roe`, `roa`, `gross_profitability`, `asset_turnover`, `gross_margin`, `earnings_quality`, `accruals`, `quality_composite`); Risk (`volatility_60/252`, `downside_volatility`, `beta_252`, `idiosyncratic_volatility`, `max_drawdown_252`); Liquidity (`adv_21d`, `turnover_21d`, `amihud_illiquidity`, `zero_trading_days`); Size (`log_market_cap`). |
| **Preprocessing** | Winsorize, z-score, rank, industry-neutralise, size-neutralise, missing treatment — all config-driven (`configs/default.yaml`). |
| **Factor eval** | Pearson/Rank IC, ICIR, positive-IC ratio, factor turnover, quantile portfolio returns, long-short spread, factor decay, cumulative factor return, monthly/annual stability; IC time-series, IC distribution, quantile-return, decay and correlation-heatmap charts; per-factor tear sheet. |
| **ML Alpha** | Cross-sectional expected-return / ranking models: `Ridge`, `ElasticNet`, `RandomForest`, `LightGBM`. Walk-forward CV with **purge + embargo** (expanding/rolling), no `shuffle=True`. Metrics: Rank-IC, ICIR, Spearman, top-quantile return, long-short, turnover, yearly stability, feature importance. |
| **Portfolio** | `PortfolioOptimizer` (CVXPY/CLARABEL): `equal_weight`, `min_variance`, `mean_variance`, `max_sharpe`, `risk_parity`. Constraints: long-only, fully-invested, max/min weight, industry limits, target volatility, turnover limit, cash buffer, max holdings. |
| **Risk** | Multi-factor (Barra-style) model `Σ = B F Bᵀ + D`: market + industry + style (`size/value/momentum/volatility/liquidity/quality`) exposures, covariance, specific variance, portfolio vol, **Euler marginal & component risk contributions** (sum to total variance). Covariance: sample, Ledoit-Wolf shrinkage, EWMA. Market-regime split (Bull/Bear × High/Low-Vol). Stress testing (historical + factor shocks). |
| **Backtest** | Event/vectorised hybrid engine. Explicit signal date → execution date (`execution_lag_days`) → rebalance date. Handles positions, cash, turnover, commission, bid-ask, slippage, market impact, corporate actions (force-liquidate delisted names — no silent survivorship bias), benchmark. Daily/weekly/monthly rebalance. |
| **Costs** | `CostModel`: commission + bid-ask + slippage + market impact (participation-rate), all parameterised. Outputs gross vs **net** (pre/post-cost) return, Sharpe, CAGR, cost drag. |
| **Performance** | CAGR, ann vol, Sharpe, Sortino, Calmar, MaxDD, beta, alpha, IR, TE, VaR, CVaR, hit rate, turnover; performance tear sheet. |
| **Attribution** | Brinson (allocation/selection/interaction) + returns-based factor attribution (R²). |
| **API** | FastAPI + Pydantic: `/health`, `/portfolio/*`, `/factors`, `/factors/{name}`, `POST /backtests`, `POST /optimize`, `POST /agent/query`, `/research/run`, `/research/last`. Input validation, error handling, logging. |
| **Dashboard** | Streamlit, Bloomberg-terminal-style dark theme: Overview, Portfolio, Factor Research, Risk, Attribution, Backtest (parameterised), Research Copilot. |
| **Copilot** | `agents/tools.py` exposes 10 real analytical tools (`factors`, `model`, `backtest`, `diagnostics`, `risk`, `attribution`, `regime`, `stress`, `quality`, `config`). The LLM answers **only** from tool outputs — it never invents reasons. LLM is optional (tool layer runs without it). |
| **Eng** | `pyproject.toml`, `Makefile`, `docker-compose.yml`, `Dockerfile`, `.pre-commit`, GitHub Actions CI (py3.10/3.11/3.12 + integration), `docs/` (mermaid architecture). |

---

## 4. Quant Methodology

- **Factors** are computed on *adjusted* prices; forward returns lag by the
  execution lag so a signal dated `t` can only be traded at `t + lag`.
- **ML** predicts cross-sectional expected return / rank, not "up/down". Trained
  under walk-forward CV with purge (drop last `H` train rows) and embargo (gap
  after train) so labels that are still forming cannot leak.
- **Portfolio** maximises `μᵀw − λ·wᵀΣw` (mean-variance) or `μᵀw/√(wᵀΣw)`
  (max-Sharpe) under long-only + weight + industry + turnover constraints via
  CVXPY.
- **Risk** decomposes variance with Euler attribution: `contrib_k =
  exposure_k · (F·exposure)_k`, which sums exactly to total variance.
- **Backtest** earns the day's return on the *pre-trade* book and deducts costs
  from NAV at session end, so costs genuinely depress the net return series
  (this is what makes gross-vs-net honest).

---

## 5. Validation — how leakage / look-ahead / survivorship are prevented

1. **Look-ahead bias** — factors and forward returns use `panel.close.shift(-H)`,
   never future data. The backtest trades `execution_lag_days` sessions after the
   signal date; a signal can never be executed on the close that produced it.
2. **Leakage** — ML uses `WalkForwardSplit` (purge + embargo), never
   `train_test_split(shuffle=True)`. Factor IC is measured on out-of-sample
   forward returns.
3. **Survivorship bias** — the sample provider is a closed synthetic universe
   (no in/out). The local/yahoo/eastmoney paths explicitly document that
   current-index membership ⇒ survivorship bias; the engine force-liquidates
   names that go dark (`delist_grace_days`) instead of dropping them, avoiding an
   *undeclared* survivorship advantage.
4. **Transaction costs / slippage** — always modelled; net metrics are reported
   alongside gross.
5. **Determinism** — `set_global_seed` + fixed config; the regression suite
   includes a determinism test and was run **3× consecutively green**; the new
   benchmark test was run **10× consecutively green**.

---

## 6. Testing

- **75 tests** across `tests/unit`, `tests/integration`, `tests/regression`.
- Covers: data quality/alignment/no-leakage, factor calc + winsorisation +
  neutralisation, optimizer weight-sum/constraints/max-weight/turnover, backtest
  NAV/costs/rebalance, covariance + risk contribution, API health/core
  endpoints, Brinson identity, and **benchmark double-differencing regression**.
- CI: `ruff check`, `ruff format --check`, `pytest -m "not slow"` (py3.10/3.11/
  3.12) + full slow integration (pipeline + API). **All green** (run
  `33727301966` → success).
- This session added `tests/unit/test_benchmark_no_double_diff.py` (4 tests)
  encoding a real prior bug (see §7 / bug note).

---

## 7. Performance Results (real sample run, 2018-01-01 → 2024-12-31)

Reproduce with: `alphaforge --start 2018-01-01 --end 2024-12-31`

**Backtest** (monthly rebalance, model Rank-IC +0.0447, sample universe ~160 names):

| Metric | Net | Gross (pre-cost) |
|---|---|---|
| Total return | +5.0% | +6.2% |
| CAGR | +0.79% | +0.98% |
| Annualised vol | 13.4% | 13.4% |
| Sharpe | 0.13 | 0.14 |
| Sortino | 0.18 | 0.20 |
| Calmar | 0.035 | 0.043 |
| Max drawdown | −22.8% | −22.7% |
| Information ratio | 0.25 | — |
| Beta (vs benchmark) | 0.47 | — |
| Alpha (ann.) | +2.65% | — |
| Avg turnover | 0.24 | — |
| **Cost drag (CAGR)** | **+0.19%** | — |

**Research layer:** factor-attribution R² = 0.80 · risk-model cross-sectional
R² = 0.50 (17 factors) · Brinson active +0.02% (alloc +0.01% / sel +0.02% / inter
+0.01%) · strongest sample factors by Rank-IC: `roe` +0.048, `roa` +0.048,
`quality_composite` +0.045, `value_composite` +0.042.

> These are **modest, honest** numbers. The synthetic sample has weak,
> low-signal alpha by design, so the platform demonstrates *correct plumbing and
> honest metrics* rather than a profitable strategy. A real-data run
> (`scripts/run_real_backtest.py --provider eastmoney`) is the path to a
> production-grade backtest, but live adapters are **not validated against the
> network in this environment** (see §8).

---

## 8. Known Limitations (honest)

- **Synthetic data:** the default `sample` provider is deterministic and
  low-signal; do **not** read the §7 returns as a tradable edge.
- **Survivorship bias:** live vendors return *current* index membership;
  historical constituents are not reconstructed.
- **Fundamental release timestamps:** point-in-time fundamental timing is
  approximated, not sourced from an event feed.
- **Transaction-cost models are parametric approximations** (bps + participation
  rate), not broker-level.
- **Live vendor adapters (yahoo / eastmoney) are untested against the network**
  in this build (no market data access here); they import cleanly and degrade
  gracefully but need a real run to validate.
- **Backtested performance ≠ live trading;** historical performance is not
  indicative of future results.
- Dashboard screenshots in the README require launching the Streamlit app.

---

## 9. Repository

`https://github.com/dev-belly/alphaforge` (public). Topics:
`quantitative-finance`, `factor-investing`, `portfolio-optimization`,
`machine-learning`, `backtesting`, `risk-management`, `fintech`, `python`,
`fastapi`.

---

## 10. How to Run (quick start)

```bash
git clone https://github.com/dev-belly/alphaforge.git
cd alphaforge
make install            # pip install -e ".[api,dashboard,viz,dev]"
make test              # fast unit + regression
make lint              # ruff check + format --check
alphaforge --start 2018-01-01 --end 2024-12-31   # full sample pipeline + HTML report
make run-api           # FastAPI on :8000
make run-dashboard     # Streamlit dashboard
# optional real data:
python scripts/run_real_backtest.py --provider eastmoney --start 2018-01-01 --end 2024-12-31
```

Docker: `docker compose up` starts API + Dashboard.

---

## 11. Interview Talking Points — 20+ questions a reviewer will ask

For each, the knowledge area you should be ready to explain (the code already
implements it; learn the *why*, not just the *what*).

1. **How do you prevent look-ahead bias in factor/backtest code?**
   → `shift(-H)` forward returns, `execution_lag_days`, signal≠execution date.
2. **Why is `train_test_split(shuffle=True)` wrong for time series, and what did
   you use instead?** → walk-forward CV, purge + embargo, expanding/rolling.
3. **Explain purge and embargo in walk-forward.** → labels still forming leak;
   drop last `H` train rows, embargo gap after.
4. **How is your portfolio optimisation formulated? Show the objective.** →
   max `μᵀw − λwᵀΣw` (or max-Sharpe), constraints long-only/weight/industry/
   turnover; CVXPY/CLARABEL.
5. **Why mean-variance and not just picking top factors?** → diversification,
   covariance, concentration control.
6. **How do you estimate the covariance matrix, and why Ledoit-Wolf?** → sample
   is noisy/unstable; shrinkage pulls toward structured estimator; EWMA for
   regime.
7. **Explain the multi-factor risk model `Σ = B F Bᵀ + D`.** → market + industry
   + style exposures, specific variance; what each term captures.
8. **What is Euler risk contribution and why does it sum to total variance?**
   → `contrib_k = w_k·∂σ²/∂w_k`; marginal × weight; additivity of quadratic form.
9. **How do you attribute portfolio risk to factors vs specific?** → component
   contributions; style vs industry vs market vs specific.
10. **How are transaction costs modelled, and why report gross AND net?** →
    commission + bid-ask + slippage + market impact; costs reduce realised NAV,
    so net must be shown honestly.
11. **What is the difference between Sharpe, Sortino, Calmar, IR, TE? When to use
    each?** → excess-return/vol, downside-dev, return/MaxDD, active/TE.
12. **How do you annualise daily returns and vol, and what is the trap?** →
    ×√252, never ×√365; wrong factor inflates vol/CAGR.
13. **What is IC / Rank-IC / ICIR, and which do you trust more?** → rank-robust to
    outliers; ICIR = mean IC / std IC (stability).
14. **Why cross-sectional expected return, not "predict up/down"?** → ranking is
    more robust; classification throws away magnitude.
15. **How do you handle survivorship bias?** → force-liquidate delisted; document
    current-membership limitation for live data.
16. **What is Brinson attribution and its three terms?** → allocation + selection
    + interaction; note the two-term vs three-term approximation gap.
17. **How does the risk model deal with missing market cap / fundamentals?**
    → falls back to equal-weight / price-based style factors (eastmoney path).
18. **Explain your market-regime module.** → Bull/Bear × High/Low-Vol; per-regime
    return stats; regime-conditional factor behaviour.
19. **How does the AI copilot avoid hallucinating?** → tool-calling only; answers
    derived from real `ToolResult`s; LLM optional.
20. **What are the limitations of your backtest vs production?** → costs are
    parametric, fills are close-to-close, no intraday, no corporate-action
    micro-structure.
21. **How is the code structured for reproducibility?** → fixed seed, config-
    driven, deterministic sample data, CI on 3 Python versions.
22. **How would you make this production-grade?** → point-in-time fundamentals,
    true historical constituents, broker cost model, intraday execution,
    parameter stability / deflated-Sharpe, paper-trading validation.

---

### Bug fixed this session (regression added)

The backtest engine re-indexed `panel.benchmark` (already a *daily return*
series) and called `pct_change()` on it, double-differencing the series. That
turned a ~15%-vol benchmark into a **58,000%-vol** artefact and silently
destroyed beta, alpha, tracking error, information ratio and capture ratios.
Fixed by `ensure_returns()` (converts a price/index/NAV *level* to returns once,
but passes an already-return series through untouched), guarded by
`tests/unit/test_benchmark_no_double_diff.py`. Also pinned `ruff==0.16.5` and
fixed pre-existing F821 `Sequence` import errors so CI cannot drift, and moved a
hardcoded EastMoney token to an env var (`ALPHAFORGE_EASTMONEY_UT`).

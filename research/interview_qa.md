# Interview Q&A — AlphaForge

Twenty questions an investment-committee / quant-internship interviewer is likely
to ask about this platform, with concise, **honest** answers grounded in the
actual code. Use them to rehearse, not to memorise spin — every claim here maps
to a file in `src/alphaforge`.

> Numbers quoted are from the synthetic `sample` run (`research/case_study.md`);
> they demonstrate the *plumbing*, not a tradeable edge.

---

**1. Walk me through the architecture.**
`data → panel → factors → ML (walk-forward) → risk model → portfolio → backtest →
attribution → report → copilot`. Each stage is a pure-ish module; the CLI, the
FastAPI service and the Streamlit dashboard all call the same `run_research`
pipeline, so nothing can disagree with the report.

**2. How do you prevent look-ahead bias?**
Two structural guards. (a) The backtest executes signals `execution_lag_days`
sessions *after* the signal date (`backtest/engine.py`), so a signal can never
trade on the close that produced it. (b) Every preprocessor step is per-date and
cross-sectional, using only data available that day; factor neutralisation and IC
estimation never reach forward information.

**3. How is the walk-forward CV set up?**
`models/split.py` builds folds with **purge** (drop training observations whose
label overlaps the validation window) and **embargo** (a gap after each train
fold). The model is always evaluated out-of-sample; the IC that reaches the
portfolio is the walk-forward out-of-sample IC, never an in-sample fit.

**4. How do you handle transaction costs honestly?**
`execution/costs.py` models commission + slippage (linear in notional) + square-root
market impact; `execution/broker.py` deducts them per trade. Costs reduce the
compounding base exactly as in production, so net Sharpe/CAGR are never inflated.

**5. Why report gross vs net, and how is gross reconstructed?**
So the cost drag is explicit, not hidden. The engine adds the per-day cost drag
back into the net return series to reconstruct the pre-cost (gross) curve — exact,
no re-run. Gross and net Sharpe/CAGR/vol are shown side by side; `cost_drag_cagr`
states the gap in one number (0.19%/yr on the sample run).

**6. How do you validate factors?**
Per-date Pearson and Rank-IC with distributional stats: `ic_mean`, `icir =
ic.mean()/ic.std()`, t-stat (overlapping-window corrected), positive-IC ratio,
quantile long-short spread, year-by-year stability. Screening 42 factors at 5%
yields ~2 false positives, so a Benjamini-Hochberg FDR flag rides alongside naive
p-values (`factors/evaluation.py`).

**7. What covariance estimator do you use?**
Default `ledoit_wolf` (constant-correlation shrinkage), implemented directly so it
tolerates the ragged NaNs that delistings produce. `sample`, `ewma`, `shrinkage`
(constant-correlation target) and `factor` are also available. The `shrinkage`
method's optimal intensity is the proper Ledoit-Wolf `π̂/‖S−T‖²` with the `ρ̂`
correction.

**8. How does portfolio optimization work?**
`PortfolioConstructor` turns scores into expected returns
(`mu = shrunk_ic · z · Σ`, cash-neutral), estimates the trailing-window
covariance, then solves via `PortfolioOptimizer`: equal-weight, min-variance,
mean-variance (QP), max-sharpe (Charnes-Cooper), or risk-parity (SLSQP). Long-only,
position cap, turnover limit, industry-deviation (penalised slack) and vol-target
constraints are all explicit.

**9. What happens when the QP is infeasible?**
A relaxation ladder: drop the vol ceiling, then relax budget equality to
`sum(w) ≤ budget`. If it still fails, the engine **holds the current book**
(`fallback:<exc>` status) rather than returning a book that violates the box.

**10. How is the risk model structured?**
Fundamental multi-factor `Σ = B F Bᵀ + D` (`risk/factor_model.py`): 17 factors
(market, size, value, momentum, volatility, liquidity, quality + 10 sector
dummies) explain **R² ≈ 0.50** of cross-sectional variance; Euler risk
contributions are exact (`euler_identity_gap = 0`).

**11. How do you attribute performance?**
Brinson-Fachler by sector (allocation + selection + interaction) and returns-based
factor attribution (`attribution/`). The report shows both and the copilot states
the allocation/selection split explicitly.

**12. How is the copilot "honest" / non-hallucinating?**
It reads only a deterministic **tool layer** that wraps real upstream outputs and
returns plain objects, never prose (`agents/tools.py`). Fixed rules turn those
into a briefing; in the default `none` mode there is no LLM call at all, and the
openai/anthropic modes fall back to the deterministic brief if the call fails.

**13. How do you ensure reproducibility?**
`set_global_seed` seeds every RNG; same config + seed → same report. The copilot's
findings are rule-driven, so a reviewer can trace every sentence to a metric.

**14. How do you handle survivorship bias?**
Delisted names are force-liquidated only after a `delist_grace_days` dark window
rather than dropped — avoiding an *undeclared* survivorship bias. The provider
still supplies point-in-time membership, and the data layer carries an explicit
survivorship disclaimer.

**15. What are the limitations / where would it fail in production?**
Synthetic data is not real; costs model timing risk only partially (large-order
timing risk unmodelled); single-node in-memory API cache (needs a job queue +
object store for multi-user); `tushare` is a reserved slot with no live adapter.
These are stated in `README.md → Limitations`.

**16. How is the benchmark handled?**
Stored as a daily *return* series (ETL converts price levels exactly once).
`ensure_returns` makes that contract explicit so a return series is never
double-differenced — an earlier bug had turned a ~15%-vol benchmark into a
58,000%-vol one and silently destroyed beta/alpha.

**17. Walk me through one backtest day.**
Mark to market on the pre-trade book → earn the day's return → if a signal fired
`lag` sessions ago, the broker rebalances to target weights and deducts costs →
record end-of-day weights → generate the next signal. Untradeable names keep
their position; dark names are force-liquidated after the grace window.

**18. What's your testing / CI strategy?**
`pytest` with a `slow` marker: fast unit + regression runs gate every push; the
slow suite runs the full pipeline + live API. `ruff` (lint + format) and `mypy`
(31 findings cleared to 0) gate too. CI is green on Python 3.10/3.11/3.12.

**19. How would you productionize this?**
Add a real-time/pooled data vendor (Yahoo/AkShare adapters already exist), a job
queue + object store in front of the API, a proper execution simulator with
timing-risk modelling, and a params/experiment store. The engine itself needs
little change — it is already provider- and config-driven.

**20. Why synthetic data, and what do the sample results show?**
To demonstrate engineering without claiming a live edge. The sample run shows
CAGR 0.79% net, Sharpe 0.13, MaxDD −22.8%, cost drag 0.19%/yr, risk R² 0.50,
model Rank-IC 0.0447 — modest economic signal by design, but every pipeline stage
(factor validation → walk-forward → costs → attribution) is exercised and
reproducible.

# Portfolio Optimization — `alphaforge.portfolio`

A ranked list of alpha scores is not a portfolio. Between the two sit the
decisions that actually shape the P&L: how much risk to take, how concentrated
the book may become, how much turnover you will pay for, and how far the
portfolio may drift from the benchmark's industry profile. Those are
*constraints*, not preferences, so AlphaForge solves for them explicitly rather
than patching them on afterwards.

```python
from alphaforge.portfolio.constructor import PortfolioConstructor, ConstructionConfig

cons = PortfolioConstructor(panel, optimizer_config, construction_config)
w = cons.construct(date, scores, prev_weights=prev_w, ic=rank_ic)
```

## `PortfolioConstructor` — the glue

The constructor answers one question per rebalance date: *given the scores, the
covariance and the constraints, what should the book look like?* It is the only
place a score becomes an expected return, so the backtester can stay a pure
accounting engine.

* **Eligibility** — only names that are in the universe *and* have enough return
  history on `date` are investable (`min_observations`, default 60).
* **Covariance** — estimated on the trailing window only
  (`covariance_lookback`, default 252 trading days), via `CovarianceEstimator`
  (`ledoit_wolf` default; also `sample`, `ewma`, `shrinkage`, `factor`). Results
  are cached per date.
* **Expected returns** — `implied_expected_returns` converts the score into an
  alpha: `mu = shrunk_ic * z * sigma`, benchmark-relative (cash-neutral). The IC
  used is **always the walk-forward out-of-sample IC** from the model layer —
  never an in-sample fit — and it is shrunk toward zero (`ic_shrinkage`, default
  0.5) because an IC estimated on a few hundred cross-sections is itself noisy.
* **Volatility targeting** — if the QP returns a hotter book than the budget
  allows (e.g. a binding turnover/industry constraint), de-levering into cash is
  the honest fallback (`_apply_vol_target`).

## `PortfolioOptimizer` — the solvers

`OptimizerConfig` exposes every knob that shapes the target portfolio. Five
methods, all solved under the same constraint set:

| Method | How | Alpha required? |
|--------|-----|-----------------|
| `equal_weight` | closed form, iterative cap | no |
| `min_variance` | convex QP | no |
| `mean_variance` | convex QP: `max mu'w - 0.5*λ*w'Σw - cost*‖dw‖₁` | yes |
| `max_sharpe` | Charnes-Cooper transform of the fractional program | yes |
| `risk_parity` | equal-risk-contribution, SLSQP on the RC residual | no |

### Constraints

* `long_only` / `fully_invested` / `cash_buffer`
* `max_weight` (position cap) and `min_weight`
* `target_volatility` (vol budget, applied *last* so dust/cardinality trimming
  cannot push the book back over risk)
* `turnover_limit` (one-way, measured from the *feasible* reference book)
* `max_industry_deviation` — enforced through a **penalised slack** rather than
  a hard cap: a single-name industry cannot reach a uniform target under the
  position cap, and a hard constraint would make the problem infeasible for
  reasons that have nothing to do with risk. The slack keeps it solvable and
  charges for the excess.
* `max_holdings` — cardinality is combinatorial, so it is a **pre-screen**: the
  universe is truncated by `|alpha|` (keeping names already held first, to spend
  turnover on conviction rather than cap compliance) before the convex solve.

### Infeasibility ladder

A QP that cannot satisfy the volatility ceiling + budget + turnover limit all at
once is solved by relaxing, hardest to softest: (1) all three; (2) drop the
volatility ceiling (re-applied exactly in `_post_process`, since volatility is
homogeneous of degree one); (3) drop budget equality to `sum(w) <= budget`. A
solve that still fails is reported as `fallback:<exc>` and the engine **holds the
current book** — returning the previous weights untouched is safer than
re-imposing caps the solver could not satisfy.

### Diagnostics

`OptimizationResult.diagnostics` carries everything needed to audit the solve:
`ex_ante_vol`, `ex_ante_return`, `ex_ante_sharpe`, `effective_n`, `turnover`,
`gross_exposure`, `net_exposure`, `cash_weight`, `max_risk_contribution_share`,
and `ic_used`. Euler risk contributions are available for the risk report.

## CLI / API entry points

`optimize(mu, cov, config, **kwargs)` is the functional wrapper used by the
pipeline, CLI and the `/optimize` API endpoint. All paths converge on the same
`PortfolioOptimizer.solve`, so the dashboard, the report and the live API can
never disagree about portfolio construction.

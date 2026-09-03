# Architecture

AlphaForge is a linear pipeline of self-contained layers. Each layer consumes the
*real* outputs of the layer before it and the whole thing is orchestrated by
`alphaforge.pipeline.ResearchPipeline`. A failure in any single layer is caught
and reported, so a partial run still produces a report from the rest.

## Layers

### 1. Data (`alphaforge.data`)
A provider returns a long, canonical price/fundamental table. `DataPipeline`
persists it and hands a bundle to the next stage. The bundled `sample` provider
is fully synthetic but point-in-time, so it carries a survivorship-bias
disclaimer rather than pretending otherwise.

### 2. Panel (`alphaforge.features.panel`)
`build_panel` pivots the long table into **wide** (dates × symbols) panels with
a single, guaranteed-aligned shape: `close`, `returns`, `volume`,
`market_cap`, `industry`, `universe`. Returns are computed from *adjusted*
prices; forward returns lag by the execution window so a signal can only ever be
traded after it exists.

### 3. Factors (`alphaforge.factors`)
A registry of factor functions produces (dates × symbols) panels. A
`FactorPreprocessor` winsorizes, standardizes and industry/size-neutralizes;
`evaluate_factor` reports Rank-IC / ICIR. Neutralisation is done so two factors
never double-count the same axis.

### 4. Models (`alphaforge.models`)
`AlphaModelPipeline` runs walk-forward CV (expanding window, purge + embargo) and
returns out-of-sample predictions plus an evaluation summary. The alpha is
converted to expected returns via Grinold's fundamental law
(`implied_expected_returns`).

### 5. Risk (`alphaforge.risk`)
`FundamentalRiskModel.fit` estimates `Σ = B F Bᵀ + D` on a rolling window. The
Euler decomposition `σ_p = Σ w_i · MCR_i` is asserted in the test suite, so the
risk-contribution chart always reconciles to portfolio volatility.

### 6. Portfolio (`alphaforge.portfolio`)
`PortfolioConstructor` turns scores → expected returns → target weights. The
optimizer supports five methods; when the QP is infeasible it walks a
relaxation ladder (vol ceiling → budget → penalized industry slack) instead of
silently returning garbage.

### 7. Execution (`alphaforge.execution`)
`CostModel` prices commission + slippage + square-root market impact;
`BrokerSimulator` applies it and rescales the budget when a name is untradeable.

### 8. Backtest (`alphaforge.backtest`)
`BacktestEngine` is a pure accounting loop. Signals are generated on the
rebalance date from data available *then*; orders execute `execution_lag_days`
later. Delisted names are force-liquidated only after a grace window — dropping
them earlier would inject an undeclared survivorship bias.

### 9. Attribution (`alphaforge.attribution`)
`brinson_attribution` explains active return by *where* you were (sectors);
`factor_attribution` explains it by *what risk* you ran (styles). They are
complements, not substitutes.

### 10. Report + Copilot (`alphaforge.reporting`, `alphaforge.agents`)
`build_html` embeds every figure as base64 — one file, no external assets. The
research copilot applies fixed rules to the real tool outputs and writes a
briefing; if an LLM is configured it only prose-ifies an already-grounded brief.

## Reproducibility contract

`set_global_seed` seeds every RNG. The same config + seed → the same report. The
copilot never invents a number: every sentence traces to a metric it actually
received.

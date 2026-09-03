# Contributing to AlphaForge

Thanks for taking the time to contribute. AlphaForge is a research platform, so
the bar for a change is not "it runs" — it is "the numbers it produces can be
defended". This document explains what that means in practice.

## Getting set up

```bash
git clone https://github.com/dev-belly/alphaforge.git
cd alphaforge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dashboard,viz,dev]"
pre-commit install
```

Verify the checkout:

```bash
make test-fast      # ~30s, no heavy pipeline
make lint
```

## The five non-negotiables

Every pull request is reviewed against these. They are not stylistic
preferences; each one corresponds to a way quant backtests silently lie.

### 1. No look-ahead bias

A feature for date `t` may only use information that was **observable on or
before `t`**.

* Fundamental data enters through the point-in-time lag
  (`data.fundamental_lag_days`), never at the fiscal-period end date.
* Signals are generated on the rebalance date and executed
  `backtest.execution_lag_days` sessions later. Do not shorten this to "just
  test the idea".
* Cross-sectional transforms (winsorize, z-score, neutralize) are applied
  **row by row**, i.e. within a single date. Cross-date statistics leak.

### 2. No shuffled splits

`train_test_split(shuffle=True)` is banned on time-series data. Use
`alphaforge.models.split.WalkForwardSplitter`, which enforces expanding or
rolling windows with purge and embargo gaps.

### 3. Costs are never optional

Any new strategy path must route trades through
`alphaforge.execution.broker.BrokerSimulator`. Reports must show gross **and**
net metrics so the drag is visible.

### 4. No fabricated numbers

Never hard-code a Sharpe ratio, IC or return into a docstring, a test fixture
or the README. If a claim about performance appears anywhere, it must be
regenerated from a real run and cite the command that produced it.

### 5. Determinism

Same config + same seed → same report. If you introduce randomness, thread it
through `alphaforge.utils.config.set_global_seed` and add a regression test.

## Development workflow

```bash
make fmt            # ruff check --fix + ruff format
make lint           # ruff check + ruff format --check
make test           # full suite including the slow pipeline run
make demo           # offline end-to-end run on sample data
```

`pre-commit` runs `ruff` and `ruff-format` on staged files. The heavier checks
(mypy, pytest) are wired to the `pre-push` stage so they never block a
work-in-progress commit.

## Where things live

| Change | Goes in |
|--------|---------|
| New data source | `src/alphaforge/data/providers/` — subclass `DataProvider`, register in `providers/__init__.py` |
| New factor | `src/alphaforge/factors/` — subclass `Factor`, add to the library registry, expose via `factors/library.py` |
| New estimator | `src/alphaforge/models/estimators.py` |
| New optimiser | `src/alphaforge/portfolio/optimizer.py` + a `PortfolioMethod` enum entry |
| New risk estimator | `src/alphaforge/risk/covariance.py` |
| New endpoint | `apps/api/alphaforge_api/main.py` — add a Pydantic request/response model |
| New copilot tool | `src/alphaforge/agents/tools.py` — register in the tool registry |

Engine code lives in `src/alphaforge/`. **Never** import Streamlit from the
engine: the quant stack must stay usable headless (CLI, API, batch jobs).

## Tests

* Unit tests for pure logic: `tests/unit/`
* End-to-end runs: `tests/integration/` (mark `@pytest.mark.slow`)
* Reproducibility checks: `tests/regression/`

A new factor or metric needs at minimum:

1. a correctness test on a hand-checkable example,
2. a leakage test (shuffle the dates and assert the output degrades or the
   point-in-time guard trips).

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat(factors): add residual momentum
fix(backtest): charge costs against the post-trade NAV
docs(architecture): add risk-model data flow
test(covariance): cover Ledoit-Wolf shrinkage bounds
refactor(portfolio): extract constraint ladder
```

## Adding a data provider

1. Subclass `DataProvider` and implement at minimum `fetch_prices`.
2. Return a `MarketPanel`-compatible frame with a `DatetimeIndex` of session
   dates and columns of symbols.
3. If the source cannot supply historical index membership, set
   `survivorship_bias_disclaimer = True` — do not quietly paper over it.
4. Add an offline fixture under `tests/` so the provider is testable without
   network access.

## Reporting bugs

Please include the config (`configs/default.yaml` plus overrides), the command,
the full traceback, and the Python + package versions. A minimal reproduction
gets fixed fastest.

## Disclosure

AlphaForge is a research tool, not investment advice. Contributions must not
add marketing-style performance claims.

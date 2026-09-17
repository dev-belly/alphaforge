# Validation & limitations

## What is checked

The [CI workflow](https://github.com/dev-belly/alphaforge/actions/workflows/ci.yml)
runs lint, type checks and fast tests on Python 3.10, 3.11 and 3.12, then the full
integration suite. Read the run for the commit you are reviewing.

The [documentation workflow](https://github.com/dev-belly/alphaforge/actions/workflows/docs.yml)
builds this site with `mkdocs build --strict` on pull requests and publishes it
from `main`. The sample is generated separately by `scripts/publish_sample.py`;
a documentation build does not run the research experiment.

The [published sample](sample-run.md) records the source revision, source/configuration
hashes, dependency versions, dates, seed and output hashes. Its report and README
figures are produced from one pipeline state.

## How to read the results

- **Synthetic provider.** The shipped prices and fundamentals are generated data.
  A factor can correlate with that generator without generalizing to markets.
- **One configuration.** The example uses one seed, one sample, Ridge and a
  mean-variance portfolio. It is not a multi-seed robustness study or a comparison
  against tuned investment baselines.
- **Model scores have a scope.** Walk-forward evaluation does not make every
  descriptive factor chart out-of-sample. Inspect dates and code before drawing conclusions.
- **Execution is simulated.** Costs are modeled, liquidity constraints are
  approximations, and execution assumptions can materially change returns.
- **Universe limitations remain.** Real data adapters do not automatically supply
  a complete history of index membership, delistings and point-in-time fundamentals.
- **A report is not a live deployment.** Vendor HTTP paths, brokerage execution and
  production infrastructure have not been certified by this sample run.

## Recheck a change

```bash
pip install -e ".[api,viz,dev]"
ruff check src tests apps scripts
mypy src --ignore-missing-imports
pytest -m "not slow and not integration"
pytest --cov=alphaforge
pip install -r requirements-docs.txt
mkdocs build --strict
```

For a new sample, rerun `python scripts/publish_sample.py` and review both numbers
and manifest. HTML contains a generation timestamp: its hash records that exact
artifact, not a promise of byte-identical HTML after rerunning.

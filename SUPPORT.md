# Support

Factor research, walk-forward evaluation, portfolio construction and cost-aware backtesting.

## Start here

Read the [README](README.md) for setup, examples and method boundaries. The tested maintenance commands are defined in the [CI workflow](.github/workflows/ci.yml). Environment: Python 3.10+.

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e ".[api,dashboard,viz,dev]"
```

Reproduce first with the bundled synthetic/sample data. Real-market experiments must retain provider and availability limitations; do not infer investable performance from the sample report. Remove API keys, private datasets, broker details and identifying records before sharing.

## Ask a question or report a problem

Use [GitHub Issues](https://github.com/dev-belly/alphaforge/issues) for usage questions, reproducible bugs and scoped feature proposals. English and Chinese reports are welcome. Check existing issues first and use the bug or feature form when it fits. Support is best effort; no response time or delivery date is promised.

Include the provider, actual data source, configuration, date range, model, seed, fold coverage, execution convention and cost assumptions. A frozen input sample and redacted run manifest are more useful than a performance screenshot alone.

Include the commit SHA, operating system, dependency versions, exact command, expected behavior and a minimal reproduction. Share only redacted logs and data you are allowed to publish. Remove credentials, personal information and confidential or licensed records. For a suspected security vulnerability, follow [SECURITY.md](SECURITY.md) before posting details.

## Verify locally

Run relevant checks before reporting a regression or opening a pull request:

```bash
ruff check src apps tests scripts
ruff format --check src apps tests scripts
mypy src apps/api
python scripts/check_docs.py
pytest -m "not slow"
```

These are verification commands, not a claim that a new environment has already passed them. For changes that affect results, retain input provenance, configuration and seeds alongside the output.

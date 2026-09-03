# AlphaForge Makefile
# Common workflows for development, testing, docs and delivery.

PYTHON ?= python3
PIP    ?= pip

.PHONY: help install install-dev test test-fast lint fmt ci run serve-api dashboard docs clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (core only)
	$(PIP) install -e .

install-dev:  ## Install with api + dashboard + dev extras
	$(PIP) install -e ".[api,dashboard,viz,dev]"

test:  ## Run the full test suite (incl. slow)
	pytest

test-fast:  ## Run only the fast unit + regression suite
	pytest -m "not slow"

lint:  ## Lint with ruff
	ruff check src apps tests
	ruff format --check src apps tests

fmt:  ## Auto-format with ruff + black
	ruff check --fix src apps tests
	ruff format src apps tests

ci:  ## What CI runs
	ruff check src apps tests
	pytest -m "not slow"

run:  ## One-shot research run (writes research/reports/research_report.html)
	$(PYTHON) scripts/run_research.py --start 2016-01-01 --end 2024-12-31

serve-api:  ## Launch the FastAPI research service on :8000
	$(PYTHON) -m alphaforge.cli --serve-api --api-port 8000

dashboard:  ## Launch the Streamlit dashboard
	streamlit run apps/dashboard/streamlit_app.py

docs:  ## Serve the mkdocs documentation locally
	mkdocs serve

clean:  ## Remove caches and generated artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info

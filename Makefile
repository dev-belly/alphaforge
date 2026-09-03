# AlphaForge Makefile
# Common workflows for development, testing, docs and delivery.

PYTHON ?= python3
PIP    ?= pip

.PHONY: help install install-dev test test-fast lint fmt format ci coverage run demo serve-api run-api dashboard run-dashboard assets docs-check docker-up docker-down docs clean

.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (core only)
	$(PIP) install -e .

install-dev:  ## Install with api + dashboard + dev extras
	$(PIP) install -e ".[api,dashboard,viz,dev]"

test:  ## Run the full test suite (incl. slow)
	pytest

test-fast:  ## Run only the fast unit + regression suite
	pytest -m "not slow"

lint:  ## Lint with ruff
	ruff check src apps tests scripts
	ruff format --check src apps tests scripts

fmt:  ## Auto-format with ruff + black
	ruff check --fix src apps tests scripts
	ruff format src apps tests scripts

format: fmt  ## Alias for `make fmt`

ci:  ## Run exactly what CI's test job enforces (ruff, mypy, docs nav, fast tests)
	ruff check src apps tests scripts
	ruff format --check src apps tests scripts
	mypy src apps/api
	$(PYTHON) scripts/check_docs.py
	pytest -m "not slow"

coverage:  ## Measure line coverage of the full suite with pytest-cov
	$(PYTHON) -m pytest --cov=alphaforge --cov=apps --cov-report=term-missing

run:  ## One-shot research run (writes research/reports/research_report.html)
	$(PYTHON) scripts/run_research.py --start 2016-01-01 --end 2024-12-31

demo:  ## Zero-setup demo: offline sample data -> full pipeline -> HTML report
	$(PYTHON) -m alphaforge.cli --start 2016-01-01 --end 2024-12-31 --report-dir research/reports

serve-api:  ## Launch the FastAPI research service on :8000
	$(PYTHON) -m alphaforge.cli --serve-api --api-port 8000

run-api: serve-api  ## Alias for `make serve-api`

dashboard:  ## Launch the Streamlit dashboard
	streamlit run apps/dashboard/streamlit_app.py

run-dashboard: dashboard  ## Alias for `make dashboard`

assets:  ## Render sample-output charts into assets/ (headless; needs the viz extra)
	$(PYTHON) scripts/make_assets.py

docs-check:  ## Validate that every mkdocs.yml nav reference resolves to a non-empty file
	$(PYTHON) scripts/check_docs.py

docker-up:  ## Build and start API + dashboard with docker compose
	docker compose up --build

docker-down:  ## Stop the docker compose stack
	docker compose down

docs:  ## Serve the mkdocs documentation locally
	mkdocs serve

clean:  ## Remove caches and generated artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info

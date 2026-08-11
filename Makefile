.DEFAULT_GOAL := help
PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install data check-data test test-fast lint fmt eval eval-quick bank api review mlflow dvc-repro docker-build docker-up clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv
	$(PY) -m venv $(VENV)

install: ## Install the project and all extras
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[mlops,serve,review,dev]"

data: ## Download and extract MVTec AD (~5 GB)
	$(BIN)/python scripts/prepare_data.py

check-data: ## Verify the dataset on disk
	$(BIN)/python scripts/prepare_data.py --check-only --strict

test: ## Run the whole test suite
	$(BIN)/python -m pytest

test-fast: ## Run only tests that need neither the dataset nor the backbone
	$(BIN)/python -m pytest -m "not needs_data and not needs_backbone"

lint: ## Lint with ruff
	$(BIN)/python -m ruff check .

fmt: ## Auto-fix lint issues
	$(BIN)/python -m ruff check --fix .

eval: ## Evaluate all 15 MVTec AD categories
	$(BIN)/python -m spade.evaluate --categories all

eval-quick: ## Single-category smoke run
	$(BIN)/python -m spade.evaluate --categories bottle --run-name quick

bank: ## Build the deployable memory bank for the demo category
	$(BIN)/python scripts/build_bank.py --category bottle

api: ## Serve the FastAPI inference API on :8000
	$(BIN)/uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

review: ## Launch the Streamlit annotation review tool on :8501
	$(BIN)/streamlit run apps/streamlit/app.py

mlflow: ## Launch the MLflow UI on :5000
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

dvc-repro: ## Run the DVC pipeline end to end
	$(BIN)/dvc repro

docker-build: ## Build the API and review images
	docker compose -f docker/docker-compose.yml build

docker-up: ## Start api + review + mlflow
	docker compose -f docker/docker-compose.yml up

clean: ## Remove caches and generated run artifacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

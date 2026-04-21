.PHONY: install install-dev lint format typecheck test test-cov clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	pip install -e .

install-dev: ## Install with all dev + optional dependencies
	pip install -e ".[all]"

lint: ## Run linter (ruff)
	ruff check src/ tests/

format: ## Auto-format code (ruff)
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck: ## Run type checker (mypy)
	mypy src/autolabel3d/

test: ## Run test suite
	pytest tests/

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=autolabel3d --cov-report=term-missing --cov-report=html

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

run: ## Run the pipeline with default config (nuScenes)
	python -m autolabel3d.cli

demo: ## Run demo on a video or image
	python scripts/demo.py --input data/sample --output outputs/demo

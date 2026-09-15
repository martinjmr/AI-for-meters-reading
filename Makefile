.PHONY: help install dev-install test lint format clean

help:
	@echo "AI for Meter Reading - Available commands:"
	@echo ""
	@echo "  make install       Install package and dependencies"
	@echo "  make dev-install   Install with development dependencies"
	@echo "  make test          Run tests"
	@echo "  make lint          Run code linting (flake8)"
	@echo "  make format        Format code (black, isort)"
	@echo "  make clean         Remove build artifacts and cache"
	@echo ""

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ scripts/ tests/ --max-line-length=100
	black --check src/ scripts/ tests/
	isort --check-only src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name *.egg-info -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name .coverage -delete
	rm -rf build/ dist/

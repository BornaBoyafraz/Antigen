VENV ?= .venv
PYTHON ?= python3
VENV_PYTHON := $(VENV)/bin/python

.PHONY: install test lint typecheck eval serve bench clean

install:
	$(PYTHON) -m venv "$(VENV)"
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"

test:
	$(VENV_PYTHON) -m pytest -q

lint:
	$(VENV_PYTHON) -m ruff check .

typecheck:
	$(VENV_PYTHON) -m mypy .

eval:
	$(VENV_PYTHON) -m eval.harness

serve:
	$(VENV_PYTHON) -m uvicorn api.app:app --reload

bench:
	$(VENV_PYTHON) benchmarks/latency.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

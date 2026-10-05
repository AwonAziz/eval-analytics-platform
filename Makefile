PY ?= python
VENV := .venv
BIN := $(VENV)/Scripts
DB ?= data/warehouse/eval_analytics.duckdb
PORT ?= 8000

.PHONY: help venv install build check summary test lint api dashboard clean rebuild

help:
	@echo "make install     create the venv and install the project"
	@echo "make build       regenerate artifacts, rebuild the warehouse, run the gate"
	@echo "make check       run the quality gate against $(DB)"
	@echo "make summary     row counts and provenance"
	@echo "make test        run the test suite"
	@echo "make api         serve the SQL API on port $(PORT)"
	@echo "make dashboard   run the Streamlit dashboard"
	@echo "make report      print every analytical answer as JSON"
	@echo "make rebuild     drop the warehouse and rebuild from scratch"
	@echo "make clean       remove build output"

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

build:
	$(BIN)/python -m eval_analytics.cli --db $(DB) build

check:
	$(BIN)/python -m eval_analytics.cli --db $(DB) check

summary:
	$(BIN)/python -m eval_analytics.cli --db $(DB) summary

test:
	$(BIN)/python -m pytest -q

lint:
	$(BIN)/python -m ruff check src dashboard tests

api:
	$(BIN)/python -m eval_analytics.cli --db $(DB) serve --port $(PORT)

dashboard:
	$(BIN)/python -m streamlit run dashboard/app.py

report:
	@for r in lora-vs-full calibration quantisation drift length-buckets slice-losses; do \
		echo "=== $$r ==="; \
		$(BIN)/python -m eval_analytics.cli --db $(DB) report $$r; \
	done

rebuild:
	$(BIN)/python -m eval_analytics.cli --db $(DB) build --no-regenerate

clean:
	rm -rf data/warehouse build .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

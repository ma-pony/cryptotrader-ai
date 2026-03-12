.PHONY: install test run serve lint format pre-commit-install pre-commit-run scheduler

install:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v

run:
	arena run --pair BTC/USDT --mode paper

serve:
	arena serve --port 8003

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

scheduler:
	arena scheduler start

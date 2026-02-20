.PHONY: install test run serve lint

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

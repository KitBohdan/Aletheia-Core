.PHONY: setup test lint format api
setup:
	python -m pip install -U pip
	pip install -e . -r requirements-dev.txt
lint:
	ruff check .
	black --check .
	mypy vct

format:
	black .
test:
	pytest -q --cov=vct --cov-report=term-missing
api:
	uvicorn vct.api.app:app --reload --port 8000

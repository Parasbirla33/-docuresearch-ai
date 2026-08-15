\
.PHONY: venv install install-dev test lint typecheck run mock clean

venv:
	python -m venv .venv

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest -v

lint:
	ruff check src tests

typecheck:
	mypy src

run:
	python -m docuresearch --topic "$(TOPIC)"

mock:
	python -m docuresearch --mock

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__

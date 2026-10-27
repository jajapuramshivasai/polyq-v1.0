PYTHON ?= python3

.PHONY: build build-rust build-python clean

build: build-rust build-python

build-rust:
	cargo build --release

build-python:
	$(PYTHON) -m maturin build --release

clean:
	cargo clean
	rm -rf dist build
	find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' \) \
		-not -path './.venv/*' -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.so' -o -name '*.dylib' -o -name '*.pyd' \) \
		-not -path './.venv/*' -delete
	find . -type d -name '*.egg-info' -not -path './.venv/*' -prune -exec rm -rf {} +
	rm -f .coverage

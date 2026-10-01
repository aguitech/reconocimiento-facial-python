.PHONY: help install install-dev test lint format clean run-cli run-api docker-build docker-up docker-down demo synth

help:  ## Muestra esta ayuda.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Instala el paquete y dependencias.
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e .

install-dev:  ## Instala con dependencias de desarrollo.
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"

test:  ## Corre los tests.
	. .venv/bin/activate && pytest

test-cov:  ## Tests con cobertura.
	. .venv/bin/activate && pytest --cov=reconocimiento_facial --cov-report=term-missing

lint:  ## Lint con ruff.
	. .venv/bin/activate && ruff check src tests

format:  ## Formatea con black.
	. .venv/bin/activate && black src tests

clean:  ## Limpia artefactos.
	rm -rf build/ dist/ .pytest_cache/ .ruff_cache/ .mypy_cache/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

run-cli:  ## Muestra ayuda del CLI.
	. .venv/bin/activate && rf-detect --help

run-api:  ## Levanta el servidor.
	. .venv/bin/activate && rf-detect serve --reload

docker-build:  ## Construye imagen Docker.
	docker-compose build

docker-up:  ## Levanta el contenedor.
	docker-compose up -d

docker-down:  ## Apaga el contenedor.
	docker-compose down

synth:  ## Genera imágenes sintéticas.
	. .venv/bin/activate && python scripts/generate_synthetic_faces.py

demo:  ## Corre la demo end-to-end.
	. .venv/bin/activate && python scripts/demo.py
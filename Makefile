.PHONY: setup lint test train prune demo

setup:
	python -m pip install -U pip
	pip install -e .[dev]
	pre-commit install

lint:
	pre-commit run --all-files

test:
	pytest -q

train:
	sincpd train --config src/sincpd/configs/default.yaml

prune:
	sincpd prune --ckpt runs/diag_default/model.pt

demo:
	python scripts/demo_quickstart.py

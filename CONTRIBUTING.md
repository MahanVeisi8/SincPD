# Contributing

Thanks for considering contributing!

## Dev setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .[dev]
pre-commit install
```

Run linters & tests:
```bash
pre-commit run --all-files
pytest -q
```

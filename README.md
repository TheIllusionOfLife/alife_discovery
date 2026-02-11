# objectless_alife

Objective-free ALife PoC implementation.

## Setup

Requires Python 3.11+.

```bash
uv venv
uv sync --extra dev
```

## Run tests

```bash
uv run pytest -q
```

## Run search

```bash
uv run python -m src.run_search --phase 1 --n-rules 100 --out-dir data
```

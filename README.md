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

## Quality gates

Run the same checks as CI before opening a PR:

```bash
uv run ruff check .
uv run ruff format . --check
uv run pytest -q
```

## Run search

```bash
uv run python -m src.run_search --phase 1 --n-rules 100 --out-dir data
```

Optional dynamic filters (default off):

```bash
uv run python -m src.run_search \
  --phase 2 \
  --n-rules 100 \
  --filter-short-period \
  --short-period-max-period 2 \
  --filter-low-activity \
  --low-activity-window 5 \
  --low-activity-min-unique-ratio 0.2 \
  --out-dir data
```

## Render animation

First run a search so `data/rules/*.json` and `data/logs/*.parquet` exist.

```bash
uv run python -c "from src.visualize import render_rule_animation; \
from pathlib import Path; \
rule_json = next(Path('data/rules').glob('*.json')); \
render_rule_animation(Path('data/logs/simulation_log.parquet'), Path('data/logs/metrics_summary.parquet'), rule_json, Path('output/preview.gif'), fps=8)"
```

You can force explicit world bounds if needed:

```bash
uv run python -m src.visualize \
  --simulation-log data/logs/simulation_log.parquet \
  --metrics-summary data/logs/metrics_summary.parquet \
  --rule-json data/rules/<rule_id>.json \
  --output output/preview.gif \
  --fps 8 \
  --grid-width 20 \
  --grid-height 20
```

## Spec Coverage Matrix

- `spec.md` section 2-6: `src/world.py`, `src/rules.py`, `src/filters.py`
- `spec.md` section 7: `src/metrics.py`, `src/run_search.py`
- `spec.md` section 9: JSON + Parquet outputs in `src/run_search.py`
- `spec.md` section 10: `src/visualize.py`

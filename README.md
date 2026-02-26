# alife_discovery

Objective-free artificial life (ALife) research integrating Assembly Theory (AT) to study emergent structure without optimization targets.

## What This Repository Contains

- Deterministic, seed-driven grid-world simulation with shared rule tables
- Four observation phases for comparative experiments (density-only, density+profile, control, random walk)
- Physical inconsistency filters (halt/state-uniform) plus optional dynamic filters for ablations
- Metrics and Parquet/JSON output pipelines
- Animation rendering for inspecting individual rule trajectories

The implementation source of truth is `spec.md`.

## Quick Start

Requirements:
- Python 3.11+
- `uv`

Setup:

```bash
uv venv
uv sync --extra dev
```

Run quality checks:

```bash
uv run ruff check .
uv run ruff format . --check
uv run pytest -q
```

## Common Commands

Run a single-phase batch search:

```bash
uv run python -m alife_discovery.run_search --phase 1 --n-rules 100 --out-dir data
```

Run a two-phase experiment comparison:

```bash
uv run python -m alife_discovery.run_search \
  --experiment \
  --phases 1,2 \
  --seed-batches 3 \
  --n-rules 100 \
  --steps 200 \
  --out-dir data
```

Run a density sweep across explicit grid/agent points (both phases):

```bash
uv run python -m alife_discovery.run_search \
  --density-sweep \
  --grid-sizes 20x20,30x30 \
  --agent-counts 30,60 \
  --seed-batches 2 \
  --n-rules 100 \
  --steps 200 \
  --out-dir data
```

Render visualizations (subcommands: `single`, `batch`, `figure`, `filmstrip`):

```bash
uv run python -m alife_discovery.visualize single \
  --simulation-log data/logs/simulation_log.parquet \
  --metrics-summary data/logs/metrics_summary.parquet \
  --rule-json data/rules/<rule_id>.json \
  --output output/preview.gif \
  --fps 8

uv run python -m alife_discovery.visualize batch \
  --phase-dir P1=data/stage_b/phase_1 \
  --phase-dir P2=data/stage_b/phase_2 \
  --top-n 5 \
  --output-dir output/batch

uv run python -m alife_discovery.visualize figure \
  --p1-dir data/stage_b/phase_1 \
  --p2-dir data/stage_b/phase_2 \
  --control-dir data/stage_c/control \
  --output-dir output/figures

uv run python -m alife_discovery.visualize filmstrip \
  --simulation-log data/logs/simulation_log.parquet \
  --rule-json data/rules/<rule_id>.json \
  --output output/filmstrip.png \
  --n-frames 8
```

Run statistical significance tests:

```bash
uv run python -m alife_discovery.stats --data-dir data/stage_b
uv run python -m alife_discovery.stats --pairwise --dir-a data/stage_b --dir-b data/stage_c
```

Effect-size sign convention in stats outputs:
- `cliffs_delta` / `effect_size_r` are oriented as first-listed group minus second-listed group.
- Positive means first-listed group tends larger; negative means second-listed group tends larger.

Export web-ready JSON payloads (with optional path boundary guard via `--base-dir`):

```bash
uv run python -m alife_discovery.export_web single \
  --data-dir data/stage_d/phase_2 \
  --rule-id phase2_rs0_ss0 \
  --output output/web/single.json \
  --base-dir .

uv run python -m alife_discovery.export_web paired \
  --phase2-dir data/stage_d/phase_2 \
  --control-dir data/stage_d/control \
  --sim-seed 42 \
  --output output/web/paired.json \
  --base-dir .
```

Switch update dynamics and viability filtering directly from CLI:

```bash
uv run python -m alife_discovery.run_search \
  --phase 2 \
  --n-rules 100 \
  --update-mode synchronous \
  --no-enable-viability-filters \
  --out-dir data
```

Configuration validation is strict and fail-fast:
- invalid values (for example, non-positive `steps`, invalid filter windows/ratios)
  raise `ValueError` at config construction/CLI parse time.

## Paper

Compile the ALIFE 2026 paper (requires [tectonic](https://tectonic-typesetting.github.io/)):

```bash
tectonic paper/main.tex
```

The compiled PDF is written to `paper/main.pdf`.

## Documentation Map

- `spec.md`: canonical implementation spec
- `AGENTS.md`: agent-facing repository instructions
- `PRODUCT.md`: product intent and research goals
- `TECH.md`: stack and technical constraints
- `STRUCTURE.md`: codebase layout and conventions
- `docs/new_research_plan.md`: Assembly Theory research direction
- `docs/legacy/`: archived context/review docs kept for traceability
- `legacy/`: archived material from the prior `objectless_alife` research direction

## High-Level Architecture

- `alife_discovery/domain/world.py`: toroidal world model, agent state, collision/movement semantics
- `alife_discovery/domain/rules.py`: observation phases, indexing logic, seeded rule-table generation
- `alife_discovery/domain/filters.py`: termination and optional dynamic filter detectors
- `alife_discovery/metrics/`: post-step analysis metrics (spatial, temporal, information submodules)
- `alife_discovery/experiments/search.py`: batch/experiment runner + artifact persistence
- `alife_discovery/analysis/stats.py`: statistical significance testing (Mann-Whitney U, chi-squared, effect sizes)
- `alife_discovery/viz/cli.py`: visualization CLI parsing and subcommand dispatch
- `alife_discovery/viz/render.py`: visualization rendering and figure generation logic
- `alife_discovery/viz/export_web.py`: web export utility for paired/single visualization payloads

## Data Outputs

By default, runs produce:
- `data/rules/*.json`: per-rule metadata, filter outcomes, seeds
- `data/logs/simulation_log.parquet`: per-agent per-step state/action logs
- `data/logs/metrics_summary.parquet`: per-step metric summaries
- `data/logs/experiment_runs.parquet`: per-rule aggregate outcomes (experiment mode)
- `data/logs/phase_summary.parquet`: per-phase aggregates (experiment mode)
- `data/logs/phase_comparison.json`: phase delta summary (experiment mode)
- `data/logs/density_sweep_runs.parquet`: per-rule aggregate outcomes (density sweep mode)
- `data/logs/density_phase_summary.parquet`: per-density/per-phase aggregates (density sweep mode)
- `data/logs/density_phase_comparison.parquet`: phase deltas for each density point (density sweep mode)

## Development Workflow

- Branch from `main` before changes (for example `feat/<topic>`, `fix/<topic>`, `chore/<topic>`)
- Keep commits focused and imperative
- Run lint + format-check + tests locally before opening PRs
- Do not treat `docs/legacy/*` as normative when it conflicts with `spec.md`

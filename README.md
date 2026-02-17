# objectless_alife

Objective-free artificial life (ALife) proof-of-concept for exploring emergent structure without optimization targets.

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
- `tectonic` (for paper compilation)

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

Compile the paper:

```bash
tectonic paper/main.tex
tectonic paper/supplementary.tex
```

## Common Commands

Run a single-phase batch search:

```bash
uv run python -m objectless_alife.run_search --phase 1 --n-rules 100 --out-dir data
```

Run a two-phase experiment comparison:

```bash
uv run python -m objectless_alife.run_search \
  --experiment \
  --phases 1,2 \
  --seed-batches 3 \
  --n-rules 100 \
  --steps 200 \
  --out-dir data
```

Run a density sweep across explicit grid/agent points (both phases):

```bash
uv run python -m objectless_alife.run_search \
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
uv run python -m objectless_alife.visualize single \
  --simulation-log data/logs/simulation_log.parquet \
  --metrics-summary data/logs/metrics_summary.parquet \
  --rule-json data/rules/<rule_id>.json \
  --output output/preview.gif \
  --fps 8

uv run python -m objectless_alife.visualize batch \
  --phase-dir P1=data/stage_b/phase_1 \
  --phase-dir P2=data/stage_b/phase_2 \
  --top-n 5 \
  --output-dir output/batch

uv run python -m objectless_alife.visualize figure \
  --p1-dir data/stage_b/phase_1 \
  --p2-dir data/stage_b/phase_2 \
  --control-dir data/stage_c/control \
  --output-dir output/figures

uv run python -m objectless_alife.visualize filmstrip \
  --simulation-log data/logs/simulation_log.parquet \
  --rule-json data/rules/<rule_id>.json \
  --output output/filmstrip.png \
  --n-frames 8
```

Run statistical significance tests:

```bash
uv run python -m objectless_alife.stats --data-dir data/stage_b
uv run python -m objectless_alife.stats --pairwise --dir-a data/stage_b --dir-b data/stage_c
```

Run follow-up heavy-compute analyses from PR #26:

```bash
uv run python scripts/no_filter_analysis.py --out-dir data/post_hoc/no_filter
uv run python scripts/synchronous_ablation.py --out-dir data/post_hoc/synchronous_ablation
uv run python scripts/ranking_stability.py --out-dir data/post_hoc/ranking_stability
uv run python scripts/te_null_analysis.py --data-dir data/stage_d --out-dir data/post_hoc/te_null
uv run python scripts/phenotype_taxonomy.py --data-dir data/stage_d --out-dir data/post_hoc/phenotypes
```

Switch update dynamics and viability filtering directly from CLI:

```bash
uv run python -m objectless_alife.run_search \
  --phase 2 \
  --n-rules 100 \
  --update-mode synchronous \
  --no-enable-viability-filters \
  --out-dir data
```

## Documentation Map

- `spec.md`: canonical implementation spec
- `AGENTS.md`: agent-facing repository instructions
- `PRODUCT.md`: product intent and research goals
- `TECH.md`: stack and technical constraints
- `STRUCTURE.md`: codebase layout and conventions
- `docs/stage_b_results.md`: Stage B experimental results
- `docs/stage_c_results.md`: Stage C experimental results
- `docs/legacy/`: archived context/review docs kept for traceability
- `paper/`: ALIFE conference paper draft (LaTeX) and figures

## High-Level Architecture

- `objectless_alife/world.py`: toroidal world model, agent state, collision/movement semantics
- `objectless_alife/rules.py`: observation phases, indexing logic, seeded rule-table generation
- `objectless_alife/filters.py`: termination and optional dynamic filter detectors
- `objectless_alife/metrics.py`: post-step analysis metrics
- `objectless_alife/run_search.py`: batch/experiment runner + artifact persistence
- `objectless_alife/stats.py`: statistical significance testing (Mann-Whitney U, chi-squared, effect sizes)
- `objectless_alife/visualize.py`: visualization compatibility entrypoint (`python -m objectless_alife.visualize`)
- `objectless_alife/viz_cli.py`: visualization CLI parsing and subcommand dispatch
- `objectless_alife/viz_render.py`: visualization rendering and figure generation logic
- `objectless_alife/export_web.py`: web export utility for paired/single visualization payloads

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

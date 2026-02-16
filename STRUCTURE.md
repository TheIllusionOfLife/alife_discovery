# STRUCTURE.md

## Repository Layout

- `spec.md`: canonical implementation specification
- `README.md`: human-facing quickstart and usage
- `AGENTS.md`: agent-specific implementation/etiquette guidance
- `PRODUCT.md`: product intent and user/value framing
- `TECH.md`: stack and constraints
- `STRUCTURE.md`: this document
- `docs/legacy/`: archived historical proposal/review docs
- `docs/stage_b_results.md`: Stage B experimental results
- `docs/stage_c_results.md`: Stage C experimental results
- `paper/`: ALIFE conference paper draft (LaTeX) and figures
- `objectless_alife/`: application source modules (Python package)
- `tests/`: test modules mirroring `objectless_alife/`
- `.github/workflows/`: CI and automation workflows

## Source Module Responsibilities

- `objectless_alife/world.py`: world model, agents, movement/collision mechanics, step updates
- `objectless_alife/rules.py`: observation phase enums, table indexing, seeded rule generation
- `objectless_alife/filters.py`: termination/dynamic filter detectors
- `objectless_alife/metrics.py`: simulation analysis metrics
- `objectless_alife/schemas.py`: Parquet schemas, schema version constants, metric-name lists
- `objectless_alife/config.py`: frozen configuration dataclasses (`SearchConfig`, `ExperimentConfig`, etc.) and safety constants
- `objectless_alife/simulation.py`: core simulation engine (`run_batch_search`) and per-step metric computation
- `objectless_alife/aggregation.py`: experiment orchestration (`run_experiment`, `run_density_sweep`, `run_multi_seed_robustness`, `run_halt_window_sweep`) and aggregation helpers
- `objectless_alife/run_search.py`: thin CLI entrypoint with `--config` JSON file support; re-exports all public symbols for backward compatibility
- `objectless_alife/stats.py`: statistical significance testing, pairwise comparisons, effect sizes
- `objectless_alife/visualize.py`: animation rendering from stored artifacts; supports `--theme` preset selection
- `objectless_alife/viz_theme.py`: visualization theme presets (`Theme` dataclass, `default` and `paper` themes)

## Test Organization

- Test files follow `tests/test_<module>.py`
- Each source module has a corresponding test module
- Prefer deterministic tests with explicit seeds and focused behavior assertions

## Naming Conventions

- Files/functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

## Import And Dependency Patterns

- Keep imports explicit by module responsibility
- Avoid cross-module leakage of concerns (for example, metrics logic in world module)
- Keep utility functions near their owning domain unless shared by multiple modules
- `run_search.py` re-exports all public symbols for backward compatibility

## Directory Hygiene Rules

- Root should contain only active project entry docs/configs
- Generated artifacts go to `data/` and `output/` and remain untracked
- Historical/context-only docs belong under `docs/legacy/`

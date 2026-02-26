# CLAUDE.md — alife_discovery

## Research Context

This repository explores **objective-free artificial life (ALife)** through the lens of **Assembly Theory (AT)**. The goal is to discover and characterize emergent structure in multi-agent grid-world simulations without optimization targets, targeting the ALIFE conference.

Prior research direction (`objectless_alife`) is archived under `legacy/`.

## Python Package

- Package name: `alife_discovery` (was `objectless_alife` before the Assembly Theory pivot)
- Import as: `from alife_discovery.domain.rules import ObservationPhase`

## Tooling

- **Python**: `uv` for package management and virtual envs (`uv run`, `uv sync --extra dev`)
- **Linting/formatting**: `ruff` (`uv run ruff check .`, `uv run ruff format . --check`)
- **Type checking**: `mypy` (`uv run mypy`)
- **Tests**: `pytest` (`uv run pytest -q`)
- **Paper**: `tectonic` for LaTeX compilation (`tectonic paper/main.tex`)

Never install packages globally. Always use `uv run` for commands.

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `alife_discovery/` | Main Python package (layered subpackages) |
| `tests/` | Test suite (mirrors source structure) |
| `scripts/` | Analysis and utility scripts (active research) |
| `docs/` | Research documentation |
| `paper/` | ALIFE 2026 paper source (LaTeX, figures, bib) |
| `legacy/` | **Archived** — prior `objectless_alife` research direction; do not import from here |

## Architecture Notes

The package is organized into focused subpackages:
- `config/`: constants and configuration dataclasses
- `domain/`: world simulation, rules, filters
- `metrics/`: spatial, temporal, information metrics
- `io/`: Parquet schemas, path helpers
- `analysis/`: statistical testing
- `simulation/`: simulation engine and persistence
- `experiments/`: search, density sweep, robustness
- `viz/`: rendering and CLI

Flat shim files (`world.py`, `rules.py`, etc.) re-export from subpackages for backward compatibility.

## Current Research Phase

**Paper writing**: ALIFE 2026 paper draft complete. Negative-result framing: assembly is entirely size-driven.

## Important Rules

- Do not import from `legacy/`
- `legacy/` is excluded from ruff and mypy checks
- Always branch before making changes (`feat/`, `fix/`, `chore/`, `docs/`)
- Never push directly to `main`

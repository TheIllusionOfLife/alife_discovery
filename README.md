# alife_discovery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19351502.svg)](https://doi.org/10.5281/zenodo.19351502)

Objective-free artificial life (ALife) research integrating Assembly Theory (AT) to study emergent structure without optimization targets.

This repository accompanies the paper:

> **Objective-Free Entity Assembly in Block Worlds: Characterizing Boundary Conditions for Emergent Complexity**
> Yuya Mukai, ALIFE 2026

## What This Repository Contains

- Block-world simulation with randomly sampled bonding rules and toroidal grid dynamics
- Assembly Theory measurement layer: exact and reuse-aware assembly index computation
- Bond-shuffle null model with empirical p-values (KS test)
- Parameter sweeps over density, grid size, and drift strength
- Partner-specific rule experiments and motif-gated catalytic positive control
- Parquet/JSON output pipelines for reproducible analysis

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

Baseline experiment (Experiment 1):

```bash
uv run python scripts/baseline_analysis.py --n-rules 100 --seeds 5 --steps 500 --out-dir data/experiment1_large
```

Assembly audit with null model (Experiment 3):

```bash
uv run python scripts/assembly_audit.py --n-rules 1000 --seeds 5 --steps 500 --n-null 100 --reuse --out-dir data/assembly_audit_v3
```

## Paper

Compile the ALIFE 2026 paper (requires [tectonic](https://tectonic-typesetting.github.io/)):

```bash
tectonic paper/main.tex
```

The compiled PDF is written to `paper/main.pdf`.

## High-Level Architecture

- `alife_discovery/domain/block_world.py`: block-world model, bonding rules, synchronous/sequential update
- `alife_discovery/domain/entity.py`: entity detection, canonicalization, WL graph hashing
- `alife_discovery/metrics/assembly.py`: exact and reuse-aware assembly index, null model
- `alife_discovery/simulation/engine.py`: batch simulation runner with Parquet persistence
- `alife_discovery/config/`: constants and configuration dataclasses
- `alife_discovery/io/`: Parquet schemas and path helpers
- `scripts/`: analysis and plotting scripts for all experiments

## Data Outputs

Experiments produce:
- `data/*/logs/entity_log.parquet`: per-entity observations with assembly metrics
- `data/*/figures/`: generated figure PDFs
- `data/*/audit_summary.txt`: summary statistics

## Citation

```bibtex
@misc{mukai2026objectivefree,
  title={Objective-Free Entity Assembly in Block Worlds: Characterizing Boundary Conditions for Emergent Complexity},
  author={Mukai, Yuya},
  year={2026},
  note={Submitted to the 2026 Conference on Artificial Life (ALIFE 2026)}
}
```

## License

Source code is licensed under [MIT](LICENSE). The paper PDF is published under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) per MIT Press
conference proceedings terms.

## Development Workflow

- Branch from `main` before changes (e.g., `feat/<topic>`, `fix/<topic>`, `chore/<topic>`)
- Keep commits focused and imperative
- Run lint + format-check + tests locally before opening PRs

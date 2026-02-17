# Reproducibility Map

This document maps manuscript outputs to canonical artifact paths.

## Canonical Dataset Lineage

- Manuscript-final four-condition analyses (Random Walk, Control, Phase 1, Phase 2): `data/stage_d/`
- Historical intermediate analyses:
  - Stage B: `data/stage_b/`
  - Stage C (control-focused): `data/stage_c/`

## Paper Asset Mapping

- `paper/figures/fig1_snapshot_grid.pdf`
  - Generated from Stage D per-phase logs under `data/stage_d/*/logs/`
- `paper/figures/fig2_mi_distribution.pdf`
  - Generated from Stage D `metrics_summary.parquet` files
- `paper/figures/fig3_mi_timeseries.pdf`
  - Generated from Stage D top-rule trajectories

## Primary Tables/Claims

- Four-condition MI / MI_excess / survival summaries in `paper/main.tex`
  - Source directories:
    - `data/stage_d/random_walk/`
    - `data/stage_d/control/`
    - `data/stage_d/phase_1/`
    - `data/stage_d/phase_2/`
- Pairwise statistical test JSON outputs:
  - `data/stage_d/logs/pairwise_ctrl_vs_p2.json`
  - `data/stage_d/logs/pairwise_p1_vs_ctrl.json`
  - `data/stage_d/logs/pairwise_p1_vs_p2.json`
  - `data/stage_d/logs/mi_excess_tests.json`

## Notes

- `data/` is generated and intentionally untracked.
- Commands in `README.md` and `AGENTS.md` are the source of truth for rerunning pipelines.

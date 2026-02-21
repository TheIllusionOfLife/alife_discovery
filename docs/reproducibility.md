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

## PR26 Follow-Up Archive

- Zenodo DOI (published on February 20, 2026): `10.5281/zenodo.18713158`
- Record URL: `https://zenodo.org/records/18713158`
- Artifact-freeze git tag: `v0.1.0-pr26-freeze`
- Current manuscript integration baseline commit (PR #36 merge): `927a1c48a8085c5479958b32cae7f0ffeab966f5`
- In-repo lightweight bundle source:
  - `data/post_hoc/pr26_followups/manifest.json`
  - `data/post_hoc/pr26_followups/checksums.sha256`
  - `data/post_hoc/pr26_followups/*/summary.json|csv`
  - `data/post_hoc/pr26_followups/phenotypes/taxonomy.json|csv`
- This archived bundle was generated with the full follow-up configuration
  (`n_rules=5000`, `steps=200`) for final post-merge PR26 reproducibility.
- Bundle integrity check command:
  - `uv run python scripts/verify_pr26_followups_bundle.py --followup-dir data/post_hoc/pr26_followups`
- One-command reproduction entrypoint:
  - `uv run python scripts/reproduce_pr26_followups.py --mode full --data-dir data/stage_d --followup-dir data/post_hoc/pr26_followups --with-paper`
- Reviewer-response mapping note:
  - `docs/reviewer_response_mapping.md`

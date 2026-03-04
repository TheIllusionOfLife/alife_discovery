# Anonymous Paper Reproducibility Guide

This guide reproduces the claims in `paper/main.tex` while preserving double-blind anonymity.

## Scope

- Paper claims are produced from the **block-world Assembly Theory track**.
- Do not use `alife_discovery.run_search` phase-track outputs for paper figures.

## Environment

```bash
uv venv
uv sync --extra dev
```

## Core Runs (paper scale)

```bash
uv run python scripts/baseline_analysis.py \
  --n-rules 1000 \
  --seeds 5 \
  --steps 500 \
  --out-dir data/experiment1_large

uv run python scripts/assembly_audit.py \
  --n-rules 1000 \
  --seeds 5 \
  --steps 500 \
  --n-null 100 \
  --reuse \
  --write-timeseries \
  --out-dir data/assembly_audit_v3

uv run python scripts/hierarchical_analysis.py \
  --input data/assembly_audit_v3/entity_log_combined.parquet \
  --out-dir data/hierarchical_analysis
```

## Figures

```bash
uv run python scripts/plot_baseline.py \
  --in-file data/experiment1_large/entity_log_combined.parquet \
  --out-dir paper/figures

uv run python scripts/plot_assembly_audit.py \
  --in-file data/assembly_audit_v3/entity_log_combined.parquet \
  --out-dir paper/figures
```

## Integrity Checks

- Use `docs/paper_result_manifest.json` to verify claim-to-artifact mapping.
- Use `docs/zenodo_upload_manifest.md` and `docs/checksums_sha256.txt` for artifact checksums.

## Double-Blind Safety

- Keep data/code availability in the manuscript anonymized.
- Do not insert author identities or non-anonymous repository links into `paper/main.tex`.

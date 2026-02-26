# Artifact Publication Policy

## Publication Split

Artifact publication splits across two channels:

1. **GitHub repository** (lightweight, reviewable):
   - Paper source and compiled PDF (`paper/`)
   - Experiment scripts and analysis code (`scripts/`)
   - Compact derived outputs tracked by git (summary JSONs, small CSVs)
   - Run configs and manifests
2. **Zenodo record** (heavy, immutable, citable):
   - Combined entity log Parquet files (experiment1, assembly audit)
   - Entity gallery PDF and metadata CSV
   - Generated figure PDFs
   - Audit summary text

## What Must Not Be Committed to Git

- Raw per-seed data files (gitignored under `data/`)
- Parquet exports and gzipped archives
- Any single file or directory exceeding ~5 MB

## What Should Be Kept in Git

- Code, configs, and lightweight summaries
- Paper assets needed for review (`.tex`, figures, `.pdf`)
- Run manifests with exact parameters and seed lists
- References to archived datasets (Zenodo DOI, checksums)

## Prerequisites

- **Python**: `requests` installed (add to `pyproject.toml`)
- **`ZENODO_TOKEN`**: personal access token with `deposit:write` and
  `deposit:actions` scopes
  - Create at <https://zenodo.org/account/settings/applications/>
  - Export: `export ZENODO_TOKEN="your_token_here"` in shell profile

## Execution Runbook

### Step 1: Prepare metadata

```bash
uv run python scripts/prepare_zenodo_metadata.py \
  data/experiment1_large/entity_log_combined.parquet \
  data/entity_gallery_large/entity_gallery.pdf \
  data/entity_gallery_large/entity_gallery_meta.csv \
  data/assembly_audit_large/entity_log_combined.parquet \
  data/assembly_audit_large/audit_summary.txt \
  --experiment-name alife_discovery_large_scale \
  --steps 500 --seed-start 0 --seed-end 4 \
  --output zenodo_staging/zenodo_metadata.json
```

### Step 2: Upload to Zenodo (draft)

```bash
uv run python scripts/upload_zenodo.py \
  --metadata zenodo_staging/zenodo_metadata.json \
  --title "Objective-Free Entity Assembly in Block Worlds: Experiment Data" \
  --description "Large-scale experiment data (1000 rules x 5 seeds x 500 steps) for the ALIFE 2026 paper." \
  --version v1.0 \
  --keyword "artificial life" --keyword "assembly theory" \
  --conference-title "ALIFE 2026" \
  --language eng
```

### Step 3: Publish (after review)

```bash
uv run python scripts/upload_zenodo.py \
  --metadata zenodo_staging/zenodo_metadata.json \
  ... --publish
```

### Step 4: Update repository references

1. Update DOI in paper source (data availability section)
2. Add `@misc` entry to `paper/references.bib`
3. Commit metadata and checksum files

## Submission Sequence

```
1. Merge paper/code PR to main
2. Final "submission-ready" commit on main
3. Upload dataset to Zenodo (draft → review → publish)
4. Create GitHub Release (triggers Zenodo code archival)
5. Add DOI badges to README
6. Submit paper to venue portal
```

## Paper-Ready Checklist

- [ ] Manuscript source and compiled PDF final
- [ ] `.zenodo.json` has real author names
- [ ] Zenodo dataset record published with checksums
- [ ] Dataset DOI in paper data availability section
- [ ] Dataset entry in references.bib
- [ ] Repository toggled ON at Zenodo GitHub settings

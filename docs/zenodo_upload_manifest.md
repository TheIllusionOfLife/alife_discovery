# Zenodo Upload Manifest

Draft deposit: DOI 10.5281/zenodo.18793985

## Files to upload (v2 — paper revision)

| File | Size | Description |
|------|------|-------------|
| `data/assembly_audit_v2/entity_log_combined.parquet` | 6.3MB | Main audit: 2.8M observations with reuse AI + empirical p-values (1000 rules × 5 seeds × 200 steps, n_null=100) |
| `data/assembly_audit_v2/audit_summary.txt` | 1KB | Human-readable audit summary |
| `data/assembly_audit_v2/step_timeseries_combined.parquet` | 247KB | Step-level timeseries for stationarity analysis |
| `data/assembly_audit_v2/mechanism/mechanism_summary.txt` | 1KB | Mechanism analysis results |
| `data/catalytic_v2/baseline/entity_log_combined.parquet` | 382KB | Catalytic control baseline |
| `data/catalytic_v2/catalytic/entity_log_combined.parquet` | 398KB | Catalytic control (3× multiplier) |
| `data/catalytic_v2/catalytic_summary.txt` | 1KB | Catalytic comparison summary |
| `data/param_sweep_v2/param_sweep_summary.parquet` | 4KB | Parameter sweep summary (partial: 10×10 grid conditions) |

## Upload instructions

1. Go to https://zenodo.org/deposit/18793985
2. Upload the files listed above
3. Update the deposit metadata (description, keywords) from `.zenodo.json`
4. Save (do NOT publish until paper is accepted)

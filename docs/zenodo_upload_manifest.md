# Zenodo Upload Manifest

Draft deposit: DOI 10.5281/zenodo.18793985

## Files to upload (v3 — pre-submission fix)

| File | Size | Checksum (sha256) | Description |
|------|------|-------------------|-------------|
| `data/assembly_audit_v3/entity_log_combined.parquet` | 15MB | `7d7dfb9011e9d...` | Main audit: 7.1M observations with reuse AI + empirical p-values (1000 rules × 5 seeds × 500 steps, n_null=100) |
| `data/assembly_audit_v3/audit_summary.txt` | 1KB | `162ee1cc663a3...` | Human-readable audit summary |
| `data/assembly_audit_v3/step_timeseries_combined.parquet` | 575KB | `2bc57e32569554e...` | Step-level timeseries for stationarity analysis (250k rows) |
| `data/assembly_audit_v3/mechanism/mechanism_summary.txt` | 1KB | `cf983d175c75e...` | Mechanism analysis results |
| `data/catalytic_v2/baseline/entity_log_combined.parquet` | 382KB | `9602b7df74626...` | Catalytic control baseline |
| `data/catalytic_v2/catalytic/entity_log_combined.parquet` | 398KB | `e5337d32bd1ed...` | Catalytic control (3× multiplier) |
| `data/catalytic_v2/catalytic_summary.txt` | 1KB | `7b264621194408...` | Catalytic comparison summary |
| `data/param_sweep_v2/param_sweep_summary.parquet` | 4KB | `b6e4e46ecbc37...` | Parameter sweep summary (11 conditions: 5 density×grid + 6 density×drift) |

Full checksums are listed in `checksums_sha256.txt` for programmatic verification.

## Upload instructions

1. Go to https://zenodo.org/deposit/18793985
2. Upload the files listed above, including `checksums_sha256.txt`
3. Update the deposit metadata (description, keywords) from `.zenodo.json`
4. Save (do NOT publish until paper is accepted)

# Zenodo Upload Manifest

Draft deposit: DOI 10.5281/zenodo.18793985

## Files to upload (v2 — paper revision)

| File | Size | Checksum (sha256) | Description |
|------|------|-------------------|-------------|
| `data/assembly_audit_v2/entity_log_combined.parquet` | 6.3MB | `65a95f57b3231...` | Main audit: 2.8M observations with reuse AI + empirical p-values (1000 rules × 5 seeds × 200 steps, n_null=100) |
| `data/assembly_audit_v2/audit_summary.txt` | 1KB | `e587f519d048b...` | Human-readable audit summary |
| `data/assembly_audit_v2/step_timeseries_combined.parquet` | 247KB | `b9127577aadaf...` | Step-level timeseries for stationarity analysis |
| `data/assembly_audit_v2/mechanism/mechanism_summary.txt` | 1KB | `60fcc10c3695c...` | Mechanism analysis results |
| `data/catalytic_v2/baseline/entity_log_combined.parquet` | 382KB | `9602b7df74626...` | Catalytic control baseline |
| `data/catalytic_v2/catalytic/entity_log_combined.parquet` | 398KB | `e5337d32bd1ed...` | Catalytic control (3× multiplier) |
| `data/catalytic_v2/catalytic_summary.txt` | 1KB | `7b264621194408...` | Catalytic comparison summary |
| `data/param_sweep_v2/param_sweep_summary.parquet` | 4KB | `75729c98a87ab...` | Parameter sweep summary (partial: 10×10 grid conditions) |

Full checksums are listed in `checksums_sha256.txt` for programmatic verification.

## Upload instructions

1. Go to https://zenodo.org/deposit/18793985
2. Upload the files listed above, including `checksums_sha256.txt`
3. Update the deposit metadata (description, keywords) from `.zenodo.json`
4. Save (do NOT publish until paper is accepted)

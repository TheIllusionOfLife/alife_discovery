#!/usr/bin/env python3
"""Hierarchical statistical analysis of assembly audit data.

Addresses reviewer concern I4 (statistical independence) by:
  1. Per-unique-type excess analysis
  2. Per-run excess distribution with block bootstrap CI
  3. Clopper-Pearson one-sided upper bound at run level
  4. Detection power analysis
  5. KS test for p-value uniformity under null

Usage:
    uv run python scripts/hierarchical_analysis.py \
        --input data/assembly_audit/entity_log_combined.parquet \
        --out-dir data/hierarchical_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats as sp_stats

from alife_discovery.analysis.bootstrap import (
    bootstrap_excess_ci,
    clopper_pearson_upper,
    detection_power,
    ks_pvalue_uniformity,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchical Statistical Analysis")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/assembly_audit/entity_log_combined.parquet"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/hierarchical_analysis"))
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.input)
    n_obs = table.num_rows

    col_names = table.column_names
    if "assembly_index_null_pvalue" in col_names:
        pvalues = np.array(table.column("assembly_index_null_pvalue").to_pylist(), dtype=float)
    elif "assembly_index_null_mean" in col_names and "assembly_index_null_std" in col_names:
        # Approximate p-values from normal distribution: p ≈ sf(z) = 1 - Φ((ai - μ) / σ)
        # Uses survival function (sf) for numerical stability in the upper tail.
        ai = np.array(table.column("assembly_index").to_pylist(), dtype=float)
        null_mean = np.array(table.column("assembly_index_null_mean").to_pylist(), dtype=float)
        null_std = np.array(table.column("assembly_index_null_std").to_pylist(), dtype=float)
        # Floor for exact-zero p-values: 1/(n_shuffles+1) matches empirical shuffle resolution
        min_p = 1.0 / 101.0
        finite = np.isfinite(ai) & np.isfinite(null_mean) & np.isfinite(null_std)
        valid = finite & (null_std > 0)
        zero_std = finite & (null_std == 0)
        negative_std = finite & (null_std < 0)
        pvalues = np.ones(n_obs, dtype=float)
        z = (ai[valid] - null_mean[valid]) / null_std[valid]
        pvalues[valid] = np.clip(sp_stats.norm.sf(z), min_p, 1.0)
        # Zero variance: ai exceeds null_mean → min_p; ai <= null_mean → p=1 (already set)
        pvalues[zero_std & (ai > null_mean)] = min_p
        # Malformed rows: negative std left as p=1 (conservative) with warning
        n_neg = int(negative_std.sum())
        n_bad = int((~finite).sum())
        if n_neg > 0:
            print(f"Warning: {n_neg} rows with negative null_std treated as p=1.0")
        if n_bad > 0:
            print(f"Warning: {n_bad} rows with non-finite values treated as p=1.0")
        print(f"Note: approximated p-values from null_mean/null_std ({n_obs:,} observations)")
    else:
        raise ValueError(
            "Input must have 'assembly_index_null_pvalue' or "
            "'assembly_index_null_mean'+'assembly_index_null_std' columns"
        )
    run_ids = table.column("run_id").to_pylist()
    entity_hashes = table.column("entity_hash").to_pylist()

    if n_obs == 0:
        print("No observations found in input.")
        report_path = args.out_dir / "hierarchical_report.txt"
        report_path.write_text("No observations found in input.\n")
        return

    # 1. Per-unique-type excess
    type_pvalues: dict[str, list[float]] = {}
    for h, pv in zip(entity_hashes, pvalues, strict=True):
        type_pvalues.setdefault(h, []).append(pv)
    n_types = len(type_pvalues)
    type_excess_count = sum(
        1 for pvs in type_pvalues.values() if np.mean([p < args.alpha for p in pvs]) > 0.5
    )

    # 2. Per-run excess distribution
    run_pvalues: dict[str, list[float]] = {}
    for rid, pv in zip(run_ids, pvalues, strict=True):
        run_pvalues.setdefault(rid, []).append(pv)
    n_runs = len(run_pvalues)
    run_excess_rates = np.array(
        [float(np.mean([p < args.alpha for p in pvs])) for pvs in run_pvalues.values()]
    )

    # 3. Block bootstrap CI on overall excess rate
    boot_lo, boot_hi = bootstrap_excess_ci(run_excess_rates, n_iter=args.n_bootstrap, rng_seed=42)

    # 4. Clopper-Pearson upper bound (run-level: how many runs have any excess?)
    runs_with_excess = int((run_excess_rates > 0).sum())
    cp_upper = clopper_pearson_upper(k=runs_with_excess, n=n_runs, alpha=args.alpha)

    # 5. Detection power
    min_excess = detection_power(n_observations=n_obs, n_types=n_types, alpha=args.alpha)

    # 6. KS test for p-value uniformity
    # Filter out p-values from trivial entities (size=1, ai=0) where p=1.0 always
    ais = np.array(table.column("assembly_index").to_pylist(), dtype=int)
    nontrivial_mask = ais > 0
    if nontrivial_mask.sum() > 0:
        nontrivial_pvalues = pvalues[nontrivial_mask]
        ks_stat, ks_pval = ks_pvalue_uniformity(nontrivial_pvalues)
    else:
        ks_stat, ks_pval = 0.0, 1.0

    # Report
    lines = [
        "=== Hierarchical Statistical Analysis ===",
        "",
        f"Total observations: {n_obs:,}",
        f"Unique entity types: {n_types}",
        f"Total runs: {n_runs}",
        "",
        "--- Per-Unique-Type Excess ---",
        f"Types with majority excess (>50% obs at p<{args.alpha}): "
        f"{type_excess_count}/{n_types} ({type_excess_count / n_types * 100:.1f}%)",
        "",
        "--- Per-Run Excess Distribution ---",
        f"Mean run excess rate: {run_excess_rates.mean():.4f}",
        f"Max run excess rate: {run_excess_rates.max():.4f}",
        f"Runs with any excess: {runs_with_excess}/{n_runs}",
        "",
        "--- Block Bootstrap 95% CI ---",
        f"CI on overall excess rate: [{boot_lo:.4f}, {boot_hi:.4f}]",
        "",
        "--- Clopper-Pearson Upper Bound ---",
        f"Upper bound on run-level excess rate (95%): {cp_upper:.4f}",
        f"Interpretation: if run-level excess exists, it is < {cp_upper:.1%} with 95% confidence",
        "",
        "--- Detection Power ---",
        f"Minimum detectable excess at 80% power: {min_excess:.4f} ({min_excess:.2%})",
        f"With {n_obs:,} observations across {n_types} types, we would detect "
        f"excess >= {min_excess:.2%} at 80% power",
        "",
        "--- P-Value Uniformity (KS Test) ---",
        f"Non-trivial entities (ai>0): {nontrivial_mask.sum():,}",
        f"KS statistic: {ks_stat:.4f}",
        f"KS p-value: {ks_pval:.4f}",
        "P-values consistent with uniformity"
        if ks_pval > 0.05
        else "P-values deviate from uniformity",
        "",
    ]

    report = "\n".join(lines)
    report_path = args.out_dir / "hierarchical_report.txt"
    report_path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()

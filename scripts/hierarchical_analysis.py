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

    pvalues = np.array(
        table.column("assembly_index_null_pvalue").to_pylist(), dtype=float
    )
    run_ids = table.column("run_id").to_pylist()
    entity_hashes = table.column("entity_hash").to_pylist()

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
    boot_lo, boot_hi = bootstrap_excess_ci(
        run_excess_rates, n_iter=args.n_bootstrap, rng_seed=42
    )

    # 4. Clopper-Pearson upper bound (run-level: how many runs have any excess?)
    runs_with_excess = int((run_excess_rates > 0).sum())
    cp_upper = clopper_pearson_upper(k=runs_with_excess, n=n_runs, alpha=args.alpha)

    # 5. Detection power
    min_excess = detection_power(
        n_observations=n_obs, n_types=n_types, alpha=args.alpha
    )

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
        f"{type_excess_count}/{n_types} ({type_excess_count/n_types*100:.1f}%)",
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
        f"Interpretation: if run-level excess exists, it is < {cp_upper:.1%} "
        f"with 95% confidence",
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

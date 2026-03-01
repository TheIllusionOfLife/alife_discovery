#!/usr/bin/env python3
"""Experiment 3: Shuffle-Bond Null Model Audit.

Runs block-world simulations with the shuffle-bond null model enabled,
then writes a combined entity log (with null columns) and an audit summary.

Usage:
    uv run python scripts/assembly_audit.py \\
        --n-rules 5 --seeds 1 --steps 20 --n-null 5 --out-dir tmp/audit_smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.simulation.engine import run_block_world_search


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3: Shuffle-Bond Null Model Audit")
    p.add_argument("--n-rules", type=int, default=100)
    p.add_argument("--seeds", type=int, default=3, help="Number of sim seeds per rule batch")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument(
        "--n-null", type=int, default=20, help="Number of null shuffles per entity (>= 1)"
    )
    p.add_argument("--grid-width", type=int, default=20)
    p.add_argument("--grid-height", type=int, default=20)
    p.add_argument("--n-blocks", type=int, default=30)
    p.add_argument("--noise-level", type=float, default=0.01)
    p.add_argument("--out-dir", type=Path, default=Path("data/assembly_audit"))
    p.add_argument("--reuse", action="store_true", help="Compute reuse-aware assembly index")
    p.add_argument("--write-timeseries", action="store_true", help="Write step-level timeseries")
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate figures after simulation via plot_assembly_audit.py",
    )
    return p.parse_args()


def _write_audit_summary(combined: pa.Table, out_path: Path) -> None:
    """Compute enrichment statistics and write a human-readable summary."""
    n = combined.num_rows
    if n == 0:
        out_path.write_text("No entity observations.\n")
        return

    ai_arr = combined.column("assembly_index").to_numpy(zero_copy_only=False).astype(float)
    nm_arr = combined.column("assembly_index_null_mean").to_numpy(zero_copy_only=False)
    ns_arr = combined.column("assembly_index_null_std").to_numpy(zero_copy_only=False)
    sz_arr = combined.column("entity_size").to_numpy(zero_copy_only=False)

    # Fraction with significant excess (a_i > null_mean + 2 * null_std)
    sig_mask = ai_arr > nm_arr + 2.0 * ns_arr
    sig_excess = int(sig_mask.sum())
    frac_sig = sig_excess / n

    # Empirical p-value significance (if column present)
    has_pvalue = "assembly_index_null_pvalue" in combined.column_names
    if has_pvalue:
        pv_arr = combined.column("assembly_index_null_pvalue").to_numpy(zero_copy_only=False)
        pv_sig = int((pv_arr < 0.05).sum())
        frac_pv = pv_sig / n

    # Mean excess overall
    excess_all = ai_arr - nm_arr
    mean_excess_overall = float(excess_all.mean())

    # Overall enrichment
    mean_observed = float(ai_arr.mean())
    mean_null = float(nm_arr.mean())

    # Reuse AI stats (if column present)
    has_reuse = "assembly_index_reuse" in combined.column_names
    if has_reuse:
        reuse_arr = (
            combined.column("assembly_index_reuse").to_numpy(zero_copy_only=False).astype(float)
        )

    # Mean excess by entity size
    unique_sizes = np.unique(sz_arr)

    lines = [
        "=== Assembly Audit Summary (Experiment 3) ===",
        f"Total entity observations: {n:,}",
        "",
        "--- Enrichment ---",
        f"Observed mean a_i:              {mean_observed:.4f}",
        f"Null mean a_i (mean of means):  {mean_null:.4f}",
        f"Overall enrichment (obs-null):  {mean_excess_overall:.4f}",
        "",
        "--- Significant Excess (a_i > null_mean + 2σ) ---",
        f"Count:    {sig_excess:,} / {n:,}",
        f"Fraction: {frac_sig:.4f} ({frac_sig * 100:.1f}%)",
    ]
    if has_pvalue:
        lines.extend(
            [
                "",
                "--- Significant Excess (empirical p < 0.05) ---",
                f"Count:    {pv_sig:,} / {n:,}",
                f"Fraction: {frac_pv:.4f} ({frac_pv * 100:.1f}%)",
            ]
        )
    if has_reuse:
        lines.extend(
            [
                "",
                "--- Reuse-Aware Assembly Index ---",
                f"Mean a_r:  {float(reuse_arr.mean()):.4f}",
                f"Max a_r:   {int(reuse_arr.max())}",
            ]
        )
    lines.extend(
        [
            "",
            "--- Mean Excess by Entity Size ---",
        ]
    )
    for sz in unique_sizes:
        mask = sz_arr == sz
        me = float(excess_all[mask].mean())
        cnt = int(mask.sum())
        lines.append(f"  size={sz}: mean_excess={me:.4f}  (n={cnt:,})")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(text)


def main() -> None:
    args = parse_args()
    if args.n_null < 1:
        sys.exit("error: --n-null must be >= 1 for assembly audit")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seeds):
        config = BlockWorldConfig(
            grid_width=args.grid_width,
            grid_height=args.grid_height,
            n_blocks=args.n_blocks,
            noise_level=args.noise_level,
            steps=args.steps,
            sim_seed=seed,
            n_null_shuffles=args.n_null,
            compute_reuse_index=args.reuse,
            write_timeseries=args.write_timeseries,
        )
        results = run_block_world_search(
            n_rules=args.n_rules,
            out_dir=args.out_dir / f"seed_{seed}",
            config=config,
        )
        print(f"Seed {seed}: {len(results)} runs complete")

    log_files = [
        args.out_dir / f"seed_{s}" / "logs" / "entity_log.parquet"
        for s in range(args.seeds)
        if (args.out_dir / f"seed_{s}" / "logs" / "entity_log.parquet").exists()
    ]
    if not log_files:
        print("No entity logs found.")
        return

    tables = [pq.read_table(f) for f in log_files]
    combined = pa.concat_tables(tables)
    out_path = args.out_dir / "entity_log_combined.parquet"
    pq.write_table(combined, out_path)
    print(f"Combined entity log: {len(combined)} rows -> {out_path}")

    summary_path = args.out_dir / "audit_summary.txt"
    _write_audit_summary(combined, summary_path)
    print(f"Audit summary: {summary_path}")

    if args.plot:
        sys.stdout.flush()
        plotter = Path(__file__).parent / "plot_assembly_audit.py"
        subprocess.run(
            [
                sys.executable,
                str(plotter),
                "--in-file",
                str(out_path),
                "--out-dir",
                str(args.out_dir / "figures"),
            ],
            check=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )


if __name__ == "__main__":
    main()

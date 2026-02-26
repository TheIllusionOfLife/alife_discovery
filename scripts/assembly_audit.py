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

import pyarrow as pa
import pyarrow.parquet as pq

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.simulation.engine import run_block_world_search


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3: Shuffle-Bond Null Model Audit")
    p.add_argument("--n-rules", type=int, default=100)
    p.add_argument("--seeds", type=int, default=3, help="Number of sim seeds per rule batch")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-null", type=int, default=20, help="Number of null shuffles per entity")
    p.add_argument("--grid-width", type=int, default=20)
    p.add_argument("--grid-height", type=int, default=20)
    p.add_argument("--n-blocks", type=int, default=30)
    p.add_argument("--noise-level", type=float, default=0.01)
    p.add_argument("--out-dir", type=Path, default=Path("data/assembly_audit"))
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate figures after simulation via plot_assembly_audit.py",
    )
    return p.parse_args()


def _write_audit_summary(combined: pa.Table, out_path: Path) -> None:
    """Compute enrichment statistics and write a human-readable summary."""
    ai = combined.column("assembly_index").to_pylist()
    null_mean_col = combined.column("assembly_index_null_mean").to_pylist()
    null_std_col = combined.column("assembly_index_null_std").to_pylist()
    sizes = combined.column("entity_size").to_pylist()

    n = len(ai)
    if n == 0:
        out_path.write_text("No entity observations.\n")
        return

    ai_arr = [float(v) for v in ai]
    nm_arr = [float(v) for v in null_mean_col]
    ns_arr = [float(v) for v in null_std_col]
    sz_arr = [int(v) for v in sizes]

    # Fraction with significant excess (a_i > null_mean + 2 * null_std)
    sig_excess = sum(1 for a, m, s in zip(ai_arr, nm_arr, ns_arr, strict=True) if a > m + 2.0 * s)
    frac_sig = sig_excess / n

    # Mean excess overall
    excess_all = [a - m for a, m in zip(ai_arr, nm_arr, strict=True)]
    mean_excess_overall = sum(excess_all) / n

    # Overall enrichment
    mean_observed = sum(ai_arr) / n
    mean_null = sum(nm_arr) / n

    # Mean excess by entity size
    size_excess: dict[int, list[float]] = {}
    for sz, exc in zip(sz_arr, excess_all, strict=True):
        size_excess.setdefault(sz, []).append(exc)

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
        "",
        "--- Mean Excess by Entity Size ---",
    ]
    for sz in sorted(size_excess):
        vals = size_excess[sz]
        me = sum(vals) / len(vals)
        lines.append(f"  size={sz}: mean_excess={me:.4f}  (n={len(vals):,})")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(text)


def main() -> None:
    args = parse_args()
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
        )
        results = run_block_world_search(
            n_rules=args.n_rules,
            out_dir=args.out_dir / f"seed_{seed}",
            config=config,
        )
        print(f"Seed {seed}: {len(results)} runs complete")

    log_files = list(args.out_dir.rglob("entity_log.parquet"))
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

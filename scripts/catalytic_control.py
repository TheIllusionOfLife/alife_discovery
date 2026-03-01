#!/usr/bin/env python3
"""Experiment: Catalytic K Positive Control.

Runs baseline (catalyst_multiplier=1.0) and catalytic conditions with same
rule seeds to compare assembly distributions.

Usage:
    uv run python scripts/catalytic_control.py \
        --n-rules 5 --seeds 1 --steps 20 --out-dir tmp/catalytic_smoke
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
    p = argparse.ArgumentParser(description="Catalytic K Positive Control")
    p.add_argument("--n-rules", type=int, default=100)

    def _positive_int(value: str) -> int:
        iv = int(value)
        if iv < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {iv}")
        return iv

    p.add_argument("--seeds", type=_positive_int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--catalyst-multiplier", type=float, default=3.0)
    p.add_argument("--n-null", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=Path("data/catalytic_control"))
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def _run_condition(
    label: str,
    args: argparse.Namespace,
    catalyst_multiplier: float,
) -> pa.Table | None:
    """Run all seeds for one condition, combine entity logs."""
    cond_dir = args.out_dir / label
    for seed in range(args.seeds):
        config = BlockWorldConfig(
            steps=args.steps,
            sim_seed=seed,
            n_null_shuffles=args.n_null,
            catalyst_multiplier=catalyst_multiplier,
            compute_reuse_index=True,
        )
        results = run_block_world_search(
            n_rules=args.n_rules,
            out_dir=cond_dir / f"seed_{seed}",
            config=config,
        )
        print(f"  {label} seed {seed}: {len(results)} runs")

    log_files = [
        cond_dir / f"seed_{s}" / "logs" / "entity_log.parquet"
        for s in range(args.seeds)
        if (cond_dir / f"seed_{s}" / "logs" / "entity_log.parquet").exists()
    ]
    if not log_files:
        return None
    tables = [pq.read_table(f) for f in log_files]
    combined = pa.concat_tables(tables)
    out_path = cond_dir / "entity_log_combined.parquet"
    pq.write_table(combined, out_path)
    print(f"  {label}: {combined.num_rows} rows -> {out_path}")
    return combined


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Baseline (catalyst_multiplier=1.0) ===")
    baseline = _run_condition("baseline", args, catalyst_multiplier=1.0)

    print(f"\n=== Catalytic (catalyst_multiplier={args.catalyst_multiplier}) ===")
    catalytic = _run_condition("catalytic", args, catalyst_multiplier=args.catalyst_multiplier)

    # Comparative summary
    if baseline is not None and catalytic is not None:
        lines = ["=== Catalytic Control Summary ===", ""]
        for label, tbl in [("Baseline", baseline), ("Catalytic", catalytic)]:
            ai = tbl.column("assembly_index").to_numpy(zero_copy_only=False).astype(float)
            sz = tbl.column("entity_size").to_numpy(zero_copy_only=False).astype(float)
            lines.append(f"{label}:")
            lines.append(f"  n_observations: {tbl.num_rows}")
            lines.append(f"  mean_ai: {ai.mean():.4f}, max_ai: {ai.max()}")
            lines.append(f"  mean_size: {sz.mean():.4f}, max_size: {sz.max()}")
            if "assembly_index_null_pvalue" in tbl.column_names:
                pv = tbl.column("assembly_index_null_pvalue").to_numpy(zero_copy_only=False)
                sig = float((pv < 0.05).mean()) * 100
                lines.append(f"  pct_excess_p05: {sig:.1f}%")
            lines.append("")

        summary_text = "\n".join(lines)
        summary_path = args.out_dir / "catalytic_summary.txt"
        summary_path.write_text(summary_text)
        print(summary_text)

    if args.plot:
        baseline_log = args.out_dir / "baseline" / "entity_log_combined.parquet"
        catalytic_log = args.out_dir / "catalytic" / "entity_log_combined.parquet"
        if not baseline_log.exists() or not catalytic_log.exists():
            print("Warning: skipping plot — combined parquet files not found")
        else:
            sys.stdout.flush()
            plotter = Path(__file__).parent / "plot_catalytic_control.py"
            subprocess.run(
                [
                    sys.executable,
                    str(plotter),
                    "--baseline-file",
                    str(baseline_log),
                    "--catalytic-file",
                    str(catalytic_log),
                    "--out-dir",
                    str(args.out_dir / "figures"),
                ],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )


if __name__ == "__main__":
    main()

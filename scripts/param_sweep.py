#!/usr/bin/env python3
"""Parameter Sweep: Density × Grid × Drift.

3-axis sweep: 4 density levels × 2 grid sizes × 4 drift probabilities.
Shows negative result robustness across parameter space.

Usage:
    uv run python scripts/param_sweep.py \
        --n-rules 5 --seeds 1 --steps 20 --out-dir tmp/sweep_smoke
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

# Density ratios: 7.5%, 15%, 30%, 50%
DENSITY_RATIOS = (0.075, 0.15, 0.30, 0.50)
GRID_SIZES = ((10, 10), (20, 20))
DRIFT_PROBABILITIES = (0.25, 0.5, 0.75, 1.0)

PARAM_SWEEP_SCHEMA = pa.schema(
    [
        ("grid_width", pa.int64()),
        ("grid_height", pa.int64()),
        ("n_blocks", pa.int64()),
        ("density_ratio", pa.float64()),
        ("drift_probability", pa.float64()),
        ("mean_ai", pa.float64()),
        ("max_ai", pa.int64()),
        ("mean_entity_size", pa.float64()),
        ("max_entity_size", pa.int64()),
        ("unique_types", pa.int64()),
        ("pct_excess_2sigma", pa.float64()),
        ("pct_excess_p05", pa.float64()),
    ]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter Sweep: Density × Grid × Drift")
    p.add_argument("--n-rules", type=int, default=100)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--n-null", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=Path("data/param_sweep"))
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def density_for_grid(ratio: float, grid_w: int, grid_h: int) -> int:
    """Compute block count for a density ratio on a given grid."""
    return max(1, int(round(ratio * grid_w * grid_h)))


def _summarize_condition(combined: pa.Table) -> dict:
    """Compute summary statistics for a single condition."""
    ai = combined.column("assembly_index").to_numpy(zero_copy_only=False).astype(float)
    sz = combined.column("entity_size").to_numpy(zero_copy_only=False).astype(float)
    unique_types = len(set(combined.column("entity_hash").to_pylist()))

    result: dict = {
        "mean_ai": float(ai.mean()),
        "max_ai": int(ai.max()),
        "mean_entity_size": float(sz.mean()),
        "max_entity_size": int(sz.max()),
        "unique_types": unique_types,
    }

    if "assembly_index_null_mean" in combined.column_names:
        nm = combined.column("assembly_index_null_mean").to_numpy(zero_copy_only=False)
        ns = combined.column("assembly_index_null_std").to_numpy(zero_copy_only=False)
        sig_2s = float((ai > nm + 2.0 * ns).mean()) * 100
        result["pct_excess_2sigma"] = sig_2s
    else:
        result["pct_excess_2sigma"] = 0.0

    if "assembly_index_null_pvalue" in combined.column_names:
        pv = combined.column("assembly_index_null_pvalue").to_numpy(zero_copy_only=False)
        sig_p = float((pv < 0.05).mean()) * 100
        result["pct_excess_p05"] = sig_p
    else:
        result["pct_excess_p05"] = 0.0

    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    # 1. Density × Grid sweep (drift=1.0)
    for gw, gh in GRID_SIZES:
        for ratio in DENSITY_RATIOS:
            n_blocks = density_for_grid(ratio, gw, gh)
            label = f"g{gw}x{gh}_d{ratio:.3f}_drift1.0"
            print(f"\n--- {label} (n_blocks={n_blocks}) ---")
            cond_dir = args.out_dir / label

            for seed in range(args.seeds):
                config = BlockWorldConfig(
                    grid_width=gw,
                    grid_height=gh,
                    n_blocks=n_blocks,
                    steps=args.steps,
                    sim_seed=seed,
                    n_null_shuffles=args.n_null,
                    drift_probability=1.0,
                )
                results = run_block_world_search(
                    n_rules=args.n_rules,
                    out_dir=cond_dir / f"seed_{seed}",
                    config=config,
                )
                print(f"  seed {seed}: {len(results)} runs")

            log_files = [
                cond_dir / f"seed_{s}" / "logs" / "entity_log.parquet"
                for s in range(args.seeds)
                if (cond_dir / f"seed_{s}" / "logs" / "entity_log.parquet").exists()
            ]
            if log_files:
                combined = pa.concat_tables([pq.read_table(f) for f in log_files])
                stats = _summarize_condition(combined)
                stats.update(
                    grid_width=gw,
                    grid_height=gh,
                    n_blocks=n_blocks,
                    density_ratio=ratio,
                    drift_probability=1.0,
                )
                summary_rows.append(stats)

    # 2. Density × Drift sweep (20×20 only, drift < 1.0)
    gw, gh = 20, 20
    for ratio in DENSITY_RATIOS:
        for drift in DRIFT_PROBABILITIES:
            if drift == 1.0:
                continue  # Already done above
            n_blocks = density_for_grid(ratio, gw, gh)
            label = f"g{gw}x{gh}_d{ratio:.3f}_drift{drift:.2f}"
            print(f"\n--- {label} (n_blocks={n_blocks}) ---")
            cond_dir = args.out_dir / label

            for seed in range(args.seeds):
                config = BlockWorldConfig(
                    grid_width=gw,
                    grid_height=gh,
                    n_blocks=n_blocks,
                    steps=args.steps,
                    sim_seed=seed,
                    n_null_shuffles=args.n_null,
                    drift_probability=drift,
                )
                results = run_block_world_search(
                    n_rules=args.n_rules,
                    out_dir=cond_dir / f"seed_{seed}",
                    config=config,
                )
                print(f"  seed {seed}: {len(results)} runs")

            log_files = [
                cond_dir / f"seed_{s}" / "logs" / "entity_log.parquet"
                for s in range(args.seeds)
                if (cond_dir / f"seed_{s}" / "logs" / "entity_log.parquet").exists()
            ]
            if log_files:
                combined = pa.concat_tables([pq.read_table(f) for f in log_files])
                stats = _summarize_condition(combined)
                stats.update(
                    grid_width=gw,
                    grid_height=gh,
                    n_blocks=n_blocks,
                    density_ratio=ratio,
                    drift_probability=drift,
                )
                summary_rows.append(stats)

    # Write summary
    if summary_rows:
        summary_table = pa.Table.from_pylist(summary_rows, schema=PARAM_SWEEP_SCHEMA)
        summary_path = args.out_dir / "param_sweep_summary.parquet"
        pq.write_table(summary_table, summary_path)
        print(f"\nSummary: {len(summary_rows)} conditions -> {summary_path}")

    if args.plot:
        sys.stdout.flush()
        plotter = Path(__file__).parent / "plot_param_sweep.py"
        subprocess.run(
            [
                sys.executable,
                str(plotter),
                "--in-file",
                str(args.out_dir / "param_sweep_summary.parquet"),
                "--out-dir",
                str(args.out_dir / "figures"),
            ],
            check=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Experiment 1: Discovery Baseline.

Run objective-free block-world rule sampling and collect (assembly_index, copy_number) data.

Usage:
    uv run python scripts/baseline_analysis.py --n-rules 10 --seeds 2 --steps 50 --out-dir tmp/smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.simulation.engine import run_block_world_search


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 1: Discovery Baseline")
    p.add_argument("--n-rules", type=int, default=100)
    p.add_argument("--seeds", type=int, default=5, help="Number of sim seeds per rule batch")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--grid-width", type=int, default=20)
    p.add_argument("--grid-height", type=int, default=20)
    p.add_argument("--n-blocks", type=int, default=30)
    p.add_argument("--noise-level", type=float, default=0.01)
    p.add_argument("--out-dir", type=Path, default=Path("data/baseline"))
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate figures after simulation via plot_baseline.py",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in range(args.seeds):
        config = BlockWorldConfig(
            grid_width=args.grid_width,
            grid_height=args.grid_height,
            n_blocks=args.n_blocks,
            noise_level=args.noise_level,
            steps=args.steps,
            sim_seed=seed,
        )
        results = run_block_world_search(
            n_rules=args.n_rules,
            out_dir=args.out_dir / f"seed_{seed}",
            config=config,
        )
        all_results.extend(results)
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

    ai_col = combined.column("assembly_index").to_pylist()
    cn_col = combined.column("copy_number_at_step").to_pylist()
    ai_mean = sum(ai_col) / len(ai_col)
    cn_mean = sum(cn_col) / len(cn_col)
    print(f"Assembly index: min={min(ai_col)}, max={max(ai_col)}, mean={ai_mean:.2f}")
    print(f"Copy number:    min={min(cn_col)}, max={max(cn_col)}, mean={cn_mean:.2f}")

    if args.plot:
        sys.stdout.flush()
        plotter = Path(__file__).parent / "plot_baseline.py"
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
        )


if __name__ == "__main__":
    main()

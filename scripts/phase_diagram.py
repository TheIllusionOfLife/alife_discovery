#!/usr/bin/env python3
"""Experiment 2: Lever Phase Diagram.

Sweep observation_range, update_mode, and noise_level and measure
P(assembly_index >= threshold, copy_number >= threshold).

Usage:
    uv run python scripts/phase_diagram.py --n-rules 20 --seeds 3 --steps 100 --out-dir tmp/phase
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alife_discovery.config.types import BlockWorldConfig, UpdateMode
from alife_discovery.simulation.engine import run_block_world_search

OBSERVATION_RANGES = [1, 2, 3]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]
UPDATE_MODES = [UpdateMode.SEQUENTIAL, UpdateMode.SYNCHRONOUS]
ASSEMBLY_THRESHOLD = 2
COPY_THRESHOLD = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2: Lever Phase Diagram")
    p.add_argument("--n-rules", type=int, default=50)
    p.add_argument("--seeds", type=int, default=3, help="Number of sim seeds per combo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-blocks", type=int, default=30)
    p.add_argument("--out-dir", type=Path, default=Path("data/phase_diagram"))
    p.add_argument("--assembly-threshold", type=int, default=ASSEMBLY_THRESHOLD)
    p.add_argument("--copy-threshold", type=int, default=COPY_THRESHOLD)
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate figures after simulation via plot_phase_diagram.py",
    )
    return p.parse_args()


def _compute_p_discovery(log_path: Path, assembly_threshold: int, copy_threshold: int) -> float:
    """Return fraction of observations meeting both thresholds."""
    if not log_path.exists():
        return 0.0
    table = pq.read_table(log_path)
    ai = table.column("assembly_index").to_numpy(zero_copy_only=False).astype(np.int64)
    cn = table.column("copy_number_at_step").to_numpy(zero_copy_only=False).astype(np.int64)
    n = len(ai)
    if n == 0:
        return 0.0
    return float(np.sum((ai >= assembly_threshold) & (cn >= copy_threshold)) / n)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    combos = list(itertools.product(OBSERVATION_RANGES, NOISE_LEVELS, UPDATE_MODES))

    for i, (obs_range, noise, update_mode) in enumerate(combos):
        label = f"obs{obs_range}_noise{noise}_{update_mode.value}"
        print(f"[{i + 1}/{len(combos)}] {label}")

        p_disc_per_seed: list[float] = []
        for seed in range(args.seeds):
            config = BlockWorldConfig(
                n_blocks=args.n_blocks,
                observation_range=obs_range,
                noise_level=noise,
                update_mode=update_mode,
                steps=args.steps,
                sim_seed=seed,
            )
            out_sub = args.out_dir / label / f"seed_{seed}"
            run_block_world_search(n_rules=args.n_rules, out_dir=out_sub, config=config)

            log_path = out_sub / "logs" / "entity_log.parquet"
            p_disc_per_seed.append(
                _compute_p_discovery(log_path, args.assembly_threshold, args.copy_threshold)
            )

        p_discovery = sum(p_disc_per_seed) / len(p_disc_per_seed)
        summary_rows.append(
            {
                "observation_range": obs_range,
                "noise_level": noise,
                "update_mode": update_mode.value,
                "p_discovery": p_discovery,
                "assembly_threshold": args.assembly_threshold,
                "copy_threshold": args.copy_threshold,
                "n_seeds": args.seeds,
            }
        )
        print(f"  P(discovery) = {p_discovery:.3f}  (avg over {args.seeds} seeds)")

    schema = pa.schema(
        [
            ("observation_range", pa.int64()),
            ("noise_level", pa.float64()),
            ("update_mode", pa.string()),
            ("p_discovery", pa.float64()),
            ("assembly_threshold", pa.int64()),
            ("copy_threshold", pa.int64()),
            ("n_seeds", pa.int64()),
        ]
    )
    out_path = args.out_dir / "phase_diagram.parquet"
    pq.write_table(pa.Table.from_pylist(summary_rows, schema=schema), out_path)
    print(f"\nPhase diagram saved: {out_path}")

    if args.plot:
        sys.stdout.flush()
        plotter = Path(__file__).parent / "plot_phase_diagram.py"
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

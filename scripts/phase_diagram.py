#!/usr/bin/env python3
"""Experiment 2: Lever Phase Diagram.

Sweep observation_range, update_mode, and noise_level and measure
P(assembly_index >= threshold, copy_number >= threshold).

Usage:
    uv run python scripts/phase_diagram.py --n-rules 20 --steps 100 --out-dir tmp/phase
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

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
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-blocks", type=int, default=30)
    p.add_argument("--out-dir", type=Path, default=Path("data/phase_diagram"))
    p.add_argument("--assembly-threshold", type=int, default=ASSEMBLY_THRESHOLD)
    p.add_argument("--copy-threshold", type=int, default=COPY_THRESHOLD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    combos = list(itertools.product(OBSERVATION_RANGES, NOISE_LEVELS, UPDATE_MODES))

    for i, (obs_range, noise, update_mode) in enumerate(combos):
        label = f"obs{obs_range}_noise{noise}_{update_mode.value}"
        print(f"[{i + 1}/{len(combos)}] {label}")

        config = BlockWorldConfig(
            n_blocks=args.n_blocks,
            observation_range=obs_range,
            noise_level=noise,
            update_mode=update_mode,
            steps=args.steps,
        )
        out_sub = args.out_dir / label
        run_block_world_search(n_rules=args.n_rules, out_dir=out_sub, config=config)

        log_path = out_sub / "logs" / "entity_log.parquet"
        if not log_path.exists():
            p_discovery = 0.0
        else:
            table = pq.read_table(log_path)
            ai = table.column("assembly_index").to_pylist()
            cn = table.column("copy_number_at_step").to_pylist()
            n = len(ai)
            p_discovery = (
                sum(
                    1
                    for a, c in zip(ai, cn, strict=True)
                    if a >= args.assembly_threshold and c >= args.copy_threshold
                )
                / n
                if n > 0
                else 0.0
            )

        summary_rows.append(
            {
                "observation_range": obs_range,
                "noise_level": noise,
                "update_mode": update_mode.value,
                "p_discovery": p_discovery,
                "assembly_threshold": args.assembly_threshold,
                "copy_threshold": args.copy_threshold,
            }
        )
        print(f"  P(discovery) = {p_discovery:.3f}")

    schema = pa.schema(
        [
            ("observation_range", pa.int64()),
            ("noise_level", pa.float64()),
            ("update_mode", pa.string()),
            ("p_discovery", pa.float64()),
            ("assembly_threshold", pa.int64()),
            ("copy_threshold", pa.int64()),
        ]
    )
    out_path = args.out_dir / "phase_diagram.parquet"
    pq.write_table(pa.Table.from_pylist(summary_rows, schema=schema), out_path)
    print(f"\nPhase diagram saved: {out_path}")


if __name__ == "__main__":
    main()

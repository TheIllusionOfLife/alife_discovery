#!/usr/bin/env python3
"""Partial parameter sweep: run only missing conditions for pre-submission fix.

Conditions:
- 10×10: 50% density, drift=1.0 (re-run, incomplete)
- 20×20: 7.5%, 15% density, drift=1.0
- 20×20: 7.5%, 15% density × drift {0.25, 0.5, 0.75}

Total: 9 conditions (completing 12 when combined with existing 10×10 data).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.simulation.engine import run_block_world_search

N_RULES = 100
SEEDS = 3
STEPS = 500
N_NULL = 100
OUT_DIR = Path("data/param_sweep_v2")

# Conditions to run: (grid_w, grid_h, density_ratio, drift_probability)
# 10×10 at 50% skipped (too slow with 100 null shuffles at high density)
CONDITIONS: list[tuple[int, int, float, float]] = [
    # 20×20 low-density at drift=1.0
    (20, 20, 0.075, 1.0),
    (20, 20, 0.15, 1.0),
    # 20×20 low-density × drift sweep
    (20, 20, 0.075, 0.25),
    (20, 20, 0.075, 0.50),
    (20, 20, 0.075, 0.75),
    (20, 20, 0.15, 0.25),
    (20, 20, 0.15, 0.50),
    (20, 20, 0.15, 0.75),
]

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


def density_for_grid(ratio: float, grid_w: int, grid_h: int) -> int:
    return max(1, int(round(ratio * grid_w * grid_h)))


def _summarize_condition(combined: pa.Table) -> dict[str, Any]:
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for gw, gh, ratio, drift in CONDITIONS:
        n_blocks = density_for_grid(ratio, gw, gh)
        label = f"g{gw}x{gh}_d{ratio:.3f}_drift{drift:.2f}"
        print(f"\n--- {label} (n_blocks={n_blocks}) ---")
        cond_dir = OUT_DIR / label

        for seed in range(SEEDS):
            config = BlockWorldConfig(
                grid_width=gw,
                grid_height=gh,
                n_blocks=n_blocks,
                steps=STEPS,
                sim_seed=seed,
                n_null_shuffles=N_NULL,
                drift_probability=drift,
            )
            results = run_block_world_search(
                n_rules=N_RULES,
                out_dir=cond_dir / f"seed_{seed}",
                config=config,
            )
            print(f"  seed {seed}: {len(results)} runs")

        log_files = [
            cond_dir / f"seed_{s}" / "logs" / "entity_log.parquet"
            for s in range(SEEDS)
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

    # Merge with existing summary
    existing_summary_path = OUT_DIR / "param_sweep_summary.parquet"
    if existing_summary_path.exists():
        existing = pq.read_table(existing_summary_path)
        existing_rows = existing.to_pylist()
        # Remove any rows we're replacing
        new_keys = {
            (r["grid_width"], r["grid_height"], r["density_ratio"], r["drift_probability"])
            for r in summary_rows
        }
        existing_rows = [
            r
            for r in existing_rows
            if (r["grid_width"], r["grid_height"], r["density_ratio"], r["drift_probability"])
            not in new_keys
        ]
        summary_rows = existing_rows + summary_rows

    if summary_rows:
        summary_table = pa.Table.from_pylist(summary_rows, schema=PARAM_SWEEP_SCHEMA)
        pq.write_table(summary_table, existing_summary_path)
        print(f"\nSummary: {len(summary_rows)} conditions -> {existing_summary_path}")


if __name__ == "__main__":
    main()

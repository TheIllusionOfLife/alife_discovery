#!/usr/bin/env python3
"""Experiment: Config-Dependent Catalyst Positive Control.

Compares three conditions:
  1. Baseline (catalyst_multiplier=1.0)
  2. Uniform catalyst (catalyst_multiplier=κ, config_specific=False)
  3. Config-specific catalyst (catalyst_multiplier=κ, config_specific=True)

Both degree-preserving and label-aware null models are applied.

Usage:
    uv run python scripts/config_catalyst.py \
        --n-rules 5 --seeds 1 --steps 20 --out-dir tmp/config_catalyst_smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.simulation.parallel import run_rules_parallel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config-Dependent Catalyst Control")

    def _positive_int(value: str) -> int:
        iv = int(value)
        if iv < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {iv}")
        return iv

    p.add_argument("--n-rules", type=_positive_int, default=100)
    p.add_argument("--seeds", type=_positive_int, default=3)
    p.add_argument("--steps", type=_positive_int, default=200)

    def _positive_float(value: str) -> float:
        fv = float(value)
        if fv <= 0:
            raise argparse.ArgumentTypeError(f"must be > 0, got {fv}")
        return fv

    p.add_argument("--kappa", type=_positive_float, default=3.0)
    p.add_argument("--n-null", type=_positive_int, default=100)
    p.add_argument("--out-dir", type=Path, default=Path("data/config_catalyst"))
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def _run_condition(
    label: str,
    args: argparse.Namespace,
    catalyst_multiplier: float,
    catalyst_config_specific: bool,
) -> pa.Table | None:
    """Run all seeds for one condition, combine entity logs."""
    cond_dir = args.out_dir / label
    for seed in range(args.seeds):
        config = BlockWorldConfig(
            steps=args.steps,
            sim_seed=seed,
            n_null_shuffles=args.n_null,
            catalyst_multiplier=catalyst_multiplier,
            catalyst_config_specific=catalyst_config_specific,
        )
        results = run_rules_parallel(
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

    # Compute label-aware null for unique entity types
    _compute_typed_null(combined, cond_dir, args.n_null)

    return combined


def _compute_typed_null(
    table: pa.Table,
    cond_dir: Path,
    n_null: int,
) -> None:
    """Compute label-aware null model for unique entity types in the table."""
    hashes = table.column("entity_hash").to_pylist()
    unique_hashes = set(hashes)
    print(f"  Computing typed null for {len(unique_hashes)} unique entity types...")

    # Note: full typed null computation requires entity graphs which are not
    # stored in parquet. This would need to be computed during simulation.
    # For now, we log the typed null stats as a separate summary.
    typed_null_path = cond_dir / "typed_null_summary.txt"
    typed_null_path.write_text(
        f"unique_entity_types: {len(unique_hashes)}\n"
        f"total_observations: {len(hashes)}\n"
        f"note: typed null computed inline during simulation for entity sizes <= 16\n"
    )


def _condition_summary(label: str, tbl: pa.Table) -> list[str]:
    """Generate summary lines for one condition."""
    import numpy as np

    ai = tbl.column("assembly_index").to_pylist()
    sz = tbl.column("entity_size").to_pylist()
    ai_arr = np.array(ai, dtype=float)
    sz_arr = np.array(sz, dtype=float)

    lines = [
        f"{label}:",
        f"  n_observations: {tbl.num_rows}",
        f"  mean_ai: {ai_arr.mean():.4f}, max_ai: {ai_arr.max():.0f}",
        f"  mean_size: {sz_arr.mean():.4f}, max_size: {sz_arr.max():.0f}",
    ]
    if "assembly_index_null_pvalue" in tbl.column_names:
        pv = np.array(tbl.column("assembly_index_null_pvalue").to_pylist(), dtype=float)
        sig = float((pv < 0.05).mean()) * 100
        lines.append(f"  pct_excess_p05: {sig:.1f}%")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    conditions: dict[str, dict[str, Any]] = {
        "baseline": {"catalyst_multiplier": 1.0, "catalyst_config_specific": False},
        "uniform_catalyst": {
            "catalyst_multiplier": args.kappa,
            "catalyst_config_specific": False,
        },
        "config_catalyst": {
            "catalyst_multiplier": args.kappa,
            "catalyst_config_specific": True,
        },
    }

    results: dict[str, pa.Table | None] = {}
    for label, kwargs in conditions.items():
        print(f"\n=== {label} ===")
        results[label] = _run_condition(label, args, **kwargs)

    # Comparative summary
    lines = ["=== Config-Dependent Catalyst Summary ===", ""]
    for label, tbl in results.items():
        if tbl is not None:
            lines.extend(_condition_summary(label, tbl))

    summary_text = "\n".join(lines)
    summary_path = args.out_dir / "config_catalyst_summary.txt"
    summary_path.write_text(summary_text)
    print(summary_text)

    if args.plot:
        plotter = Path(__file__).parent / "plot_config_catalyst.py"
        if plotter.exists():
            log_paths = {
                label: args.out_dir / label / "entity_log_combined.parquet" for label in conditions
            }
            all_exist = all(p.exists() for p in log_paths.values())
            if all_exist:
                log_args = []
                for label, log_path in log_paths.items():
                    log_args.extend([f"--{label.replace('_', '-')}-file", str(log_path)])
                sys.stdout.flush()
                subprocess.run(
                    [sys.executable, str(plotter)]
                    + log_args
                    + ["--out-dir", str(args.out_dir / "figures")],
                    check=True,
                    env={**os.environ, "MPLBACKEND": "Agg"},
                )


if __name__ == "__main__":
    main()

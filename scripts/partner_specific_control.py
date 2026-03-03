#!/usr/bin/env python3
"""Experiment: Partner-Specific Rule Expressiveness Test.

Compares standard rules (60-entry dominant-type table) vs partner-specific
rules (45-entry per-partner table) to test whether richer rule expressiveness
produces assembly excess.

Usage:
    uv run python scripts/partner_specific_control.py \
        --n-rules 5 --seeds 1 --steps 20 --out-dir tmp/partner_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.simulation.parallel import run_rules_parallel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Partner-Specific Rule Expressiveness Test")

    def _positive_int(value: str) -> int:
        iv = int(value)
        if iv < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {iv}")
        return iv

    p.add_argument("--n-rules", type=_positive_int, default=100)
    p.add_argument("--seeds", type=_positive_int, default=3)
    p.add_argument("--steps", type=_positive_int, default=200)
    p.add_argument("--n-null", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=Path("data/partner_specific"))
    return p.parse_args()


def _run_condition(
    label: str,
    args: argparse.Namespace,
    partner_specific: bool,
) -> pa.Table | None:
    """Run all seeds for one condition, combine entity logs."""
    cond_dir = args.out_dir / label
    for seed in range(args.seeds):
        config = BlockWorldConfig(
            steps=args.steps,
            sim_seed=seed,
            n_null_shuffles=args.n_null,
            partner_specific_rules=partner_specific,
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
    return combined


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Standard rules (60-entry dominant-type) ===")
    standard = _run_condition("standard", args, partner_specific=False)

    print("\n=== Partner-specific rules (45-entry per-partner) ===")
    partner = _run_condition("partner_specific", args, partner_specific=True)

    # Comparative summary
    lines = ["=== Partner-Specific Expressiveness Summary ===", ""]
    for label, tbl in [("Standard", standard), ("Partner-Specific", partner)]:
        if tbl is None:
            continue
        ai = np.array(tbl.column("assembly_index").to_pylist(), dtype=float)
        sz = np.array(tbl.column("entity_size").to_pylist(), dtype=float)
        lines.append(f"{label}:")
        lines.append(f"  n_observations: {tbl.num_rows}")
        lines.append(f"  mean_ai: {ai.mean():.4f}, max_ai: {ai.max():.0f}")
        lines.append(f"  mean_size: {sz.mean():.4f}, max_size: {sz.max():.0f}")
        if "assembly_index_null_pvalue" in tbl.column_names:
            pv = np.array(
                tbl.column("assembly_index_null_pvalue").to_pylist(), dtype=float
            )
            sig = float((pv < 0.05).mean()) * 100
            lines.append(f"  pct_excess_p05: {sig:.1f}%")
        lines.append("")

    summary_text = "\n".join(lines)
    summary_path = args.out_dir / "partner_specific_summary.txt"
    summary_path.write_text(summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()

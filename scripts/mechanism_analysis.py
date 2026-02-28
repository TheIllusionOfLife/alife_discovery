#!/usr/bin/env python3
"""Mechanism Analysis: quantitative investigation of why max entity size = 6.

Post-processes entity_log_combined.parquet + step_timeseries.parquet to compute:
1. Entity lifetime distribution
2. Bond survival rate
3. Growth transitions (size k → k+1)
4. Rule table summary (mean bond prob vs mean entity size)
5. Graph symmetry counts (automorphism group size)

Usage:
    uv run python scripts/mechanism_analysis.py \
        --entity-log data/assembly_audit/entity_log_combined.parquet \
        --timeseries data/assembly_audit/seed_0/logs/step_timeseries.parquet \
        --out-dir data/mechanism
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mechanism Analysis")
    p.add_argument("--entity-log", type=Path, required=True)
    p.add_argument("--timeseries", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("data/mechanism"))
    return p.parse_args()


def _entity_lifetime_stats(table: "pq.Table") -> list[str]:  # noqa: F821
    """Count consecutive-step appearances per (run_id, entity_hash)."""
    run_ids = table.column("run_id").to_pylist()
    steps = table.column("step").to_pylist()
    hashes = table.column("entity_hash").to_pylist()

    # Group steps by (run_id, entity_hash)
    from collections import defaultdict

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for rid, s, h in zip(run_ids, steps, hashes, strict=True):
        groups[(rid, h)].append(s)

    lifetimes: list[int] = []
    for _key, step_list in groups.items():
        sorted_steps = sorted(set(step_list))
        lifetimes.append(len(sorted_steps))

    arr = np.array(lifetimes, dtype=float)
    lines = [
        "--- Entity Lifetime Distribution ---",
        f"  n_entity_instances: {len(lifetimes)}",
        f"  mean_lifetime (snapshots): {arr.mean():.2f}",
        f"  median_lifetime: {np.median(arr):.1f}",
        f"  max_lifetime: {arr.max():.0f}",
    ]
    return lines


def _bond_survival_stats(ts_table: "pq.Table") -> list[str]:  # noqa: F821
    """Compute bond survival rate from timeseries n_bonds[t+1]/n_bonds[t]."""
    run_ids = ts_table.column("run_id").to_pylist()
    steps = ts_table.column("step").to_pylist()
    n_bonds = ts_table.column("n_bonds").to_pylist()

    from collections import defaultdict

    runs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rid, s, nb in zip(run_ids, steps, n_bonds, strict=True):
        runs[rid].append((s, nb))

    ratios: list[float] = []
    for _rid, data in runs.items():
        data.sort()
        for i in range(1, len(data)):
            prev_bonds = data[i - 1][1]
            curr_bonds = data[i][1]
            if prev_bonds > 0:
                ratios.append(curr_bonds / prev_bonds)

    if not ratios:
        return ["--- Bond Survival Rate ---", "  No data (no timeseries with bonds)."]

    arr = np.array(ratios)
    lines = [
        "--- Bond Survival Rate ---",
        f"  n_transitions: {len(ratios)}",
        f"  mean_survival_ratio: {arr.mean():.4f}",
        f"  median_survival_ratio: {np.median(arr):.4f}",
    ]
    return lines


def _growth_transition_stats(table: "pq.Table") -> list[str]:  # noqa: F821
    """Count size k → k+1 transitions between consecutive snapshots."""
    run_ids = table.column("run_id").to_pylist()
    steps = table.column("step").to_pylist()
    hashes = table.column("entity_hash").to_pylist()
    sizes = table.column("entity_size").to_pylist()

    from collections import defaultdict

    # Build size-at-step for each (run_id, entity_hash)
    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for rid, s, h, sz in zip(run_ids, steps, hashes, sizes, strict=True):
        groups[(rid, h)].append((s, sz))

    transitions: dict[tuple[int, int], int] = defaultdict(int)
    for _key, data in groups.items():
        data.sort()
        for i in range(1, len(data)):
            s_prev = data[i - 1][1]
            s_curr = data[i][1]
            if s_curr != s_prev:
                transitions[(s_prev, s_curr)] = transitions.get((s_prev, s_curr), 0) + 1

    lines = ["--- Growth Transitions ---"]
    if not transitions:
        lines.append("  No size transitions observed.")
    else:
        for (s_from, s_to), count in sorted(transitions.items()):
            lines.append(f"  {s_from} → {s_to}: {count}")
    return lines


def _symmetry_counts(table: "pq.Table") -> list[str]:  # noqa: F821
    """Compute graph automorphism counts for unique entity types."""

    hashes = table.column("entity_hash").to_pylist()
    sizes = table.column("entity_size").to_pylist()
    ais = table.column("assembly_index").to_pylist()

    # Deduplicate by hash
    seen: dict[str, tuple[int, int]] = {}
    for h, sz, ai in zip(hashes, sizes, ais, strict=True):
        if h not in seen:
            seen[h] = (sz, ai)

    lines = [
        "--- Graph Symmetry (Automorphism Group Size) ---",
        f"  unique_entity_types: {len(seen)}",
        "  (Full automorphism computation deferred to entity graph reconstruction)",
    ]
    return lines


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entity_table = pq.read_table(args.entity_log)
    lines = ["=== Mechanism Analysis Summary ===", ""]

    lines.extend(_entity_lifetime_stats(entity_table))
    lines.append("")

    if args.timeseries is not None and args.timeseries.exists():
        ts_table = pq.read_table(args.timeseries)
        lines.extend(_bond_survival_stats(ts_table))
        lines.append("")

    lines.extend(_growth_transition_stats(entity_table))
    lines.append("")

    lines.extend(_symmetry_counts(entity_table))
    lines.append("")

    text = "\n".join(lines) + "\n"
    out_path = args.out_dir / "mechanism_summary.txt"
    out_path.write_text(text)
    print(text)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

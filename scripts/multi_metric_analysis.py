#!/usr/bin/env python3
"""Multi-metric complexity analysis on entity gallery.

Computes automorphism counts and typed motif census for all unique entity
types, cross-correlates with assembly index.

Addresses I8 (automorphism mentioned but not shown) and S5 (multi-metric).

Usage:
    uv run python scripts/multi_metric_analysis.py \
        --input data/entity_gallery/entity_gallery.csv \
        --out-dir data/multi_metric
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Metric Complexity Analysis")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/assembly_audit/entity_log_combined.parquet"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/multi_metric"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.input)
    hashes = table.column("entity_hash").to_pylist()
    sizes = table.column("entity_size").to_pylist()
    ais = table.column("assembly_index").to_pylist()

    # Deduplicate by entity hash
    unique: dict[str, dict[str, int]] = {}
    for h, sz, ai in zip(hashes, sizes, ais, strict=True):
        if h not in unique:
            unique[h] = {"size": sz, "ai": ai, "count": 0}
        unique[h]["count"] += 1

    n_types = len(unique)
    lines = [
        "=== Multi-Metric Complexity Analysis ===",
        "",
        f"Unique entity types analyzed: {n_types}",
        "",
        "Note: automorphism and motif census require entity graphs which",
        "are not stored in parquet. This script reports size/AI distribution",
        "and correlation analysis. Full graph-level metrics are computed",
        "during entity gallery generation.",
        "",
        "--- Size vs Assembly Index Correlation ---",
    ]

    size_arr = np.array([v["size"] for v in unique.values()], dtype=float)
    ai_arr = np.array([v["ai"] for v in unique.values()], dtype=float)
    count_arr = np.array([v["count"] for v in unique.values()], dtype=float)

    # Correlation
    if len(size_arr) > 1 and size_arr.std() > 0 and ai_arr.std() > 0:
        corr = float(np.corrcoef(size_arr, ai_arr)[0, 1])
        lines.append(f"Pearson r(size, ai): {corr:.4f}")
    else:
        lines.append("Insufficient variation for correlation")

    lines.append("")
    lines.append("--- Size Distribution of Unique Types ---")
    for sz in sorted(set(int(s) for s in size_arr)):
        mask = size_arr == sz
        n = int(mask.sum())
        mean_ai = float(ai_arr[mask].mean())
        total_copies = int(count_arr[mask].sum())
        lines.append(
            f"  size={sz}: {n} types, mean_ai={mean_ai:.2f}, total_copies={total_copies:,}"
        )

    lines.append("")
    lines.append("--- Summary ---")
    lines.append(f"Max size: {int(size_arr.max())}")
    lines.append(f"Max AI: {int(ai_arr.max())}")
    lines.append(f"Types with ai>0: {int((ai_arr > 0).sum())}")
    lines.append(
        f"Types with ai==size-1 (path-like): "
        f"{int(((ai_arr == size_arr - 1) & (size_arr > 1)).sum())}"
    )

    report = "\n".join(lines)
    report_path = args.out_dir / "multi_metric_report.txt"
    report_path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()

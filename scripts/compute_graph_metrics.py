#!/usr/bin/env python3
"""Compute graph automorphism counts and typed motif census for entity types.

Runs block-world simulations to capture entity graphs, then computes
graph-level complexity metrics for all unique entity types discovered.

Usage:
    uv run python scripts/compute_graph_metrics.py \
        --n-rules 200 --seeds 3 --steps 200 --out-dir data/graph_metrics
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from scripts.render_entity_gallery import capture_entities

from alife_discovery.metrics.complexity import (
    graph_automorphism_count,
    typed_motif_census,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute graph metrics for entity types")
    p.add_argument("--n-rules", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--grid-width", type=int, default=20)
    p.add_argument("--grid-height", type=int, default=20)
    p.add_argument("--n-blocks", type=int, default=30)
    p.add_argument("--noise-level", type=float, default=0.0)
    p.add_argument("--out-dir", type=Path, default=Path("data/graph_metrics"))
    args = p.parse_args()

    print(f"Capturing entities: {args.n_rules} rules x {args.seeds} seeds x {args.steps} steps")
    registry = capture_entities(
        n_rules=args.n_rules,
        seeds=args.seeds,
        steps=args.steps,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        n_blocks=args.n_blocks,
        noise_level=args.noise_level,
    )

    print(f"Captured {len(registry)} unique entity types")

    # Compute metrics for each entity type
    rows: list[dict[str, object]] = []
    for h, rec in sorted(registry.items(), key=lambda x: x[1].entity_size):
        g = rec.graph
        auto_count = graph_automorphism_count(g)
        motifs = typed_motif_census(g)

        rows.append({
            "entity_hash": h,
            "entity_size": rec.entity_size,
            "assembly_index": rec.assembly_index,
            "copy_count": rec.total_copy_count,
            "automorphism_count": auto_count,
            "n_triangles": motifs["triangles"],
            "n_open_wedges": motifs["open_wedges"],
            "typed_triangles": str(motifs["typed_triangles"]),
            "typed_wedges": str(motifs["typed_wedges"]),
        })

    # Save to CSV
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "graph_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")

    # Print summary
    print(f"\n--- Graph Metrics Summary ({len(rows)} entity types) ---")
    sizes = [r["entity_size"] for r in rows]
    autos = [r["automorphism_count"] for r in rows]
    triangles = [r["n_triangles"] for r in rows]
    wedges = [r["n_open_wedges"] for r in rows]

    for sz in sorted(set(sizes)):
        subset = [r for r in rows if r["entity_size"] == sz]
        sub_autos = [r["automorphism_count"] for r in subset]
        sub_tri = [r["n_triangles"] for r in subset]
        sub_wedge = [r["n_open_wedges"] for r in subset]
        print(
            f"  Size {sz}: {len(subset)} types, "
            f"auto=[{min(sub_autos)}, {max(sub_autos)}] mean={sum(sub_autos)/len(sub_autos):.1f}, "
            f"tri=[{min(sub_tri)}, {max(sub_tri)}], "
            f"wedges=[{min(sub_wedge)}, {max(sub_wedge)}]"
        )

    # Compute Spearman correlation between size and automorphism count
    try:
        from scipy.stats import spearmanr  # type: ignore[import-untyped]

        if len(set(sizes)) > 1:
            rho, pval = spearmanr(sizes, autos)
            print(f"\nSpearman(size, automorphism_count): rho={rho:.3f}, p={pval:.4g}")
    except ImportError:
        pass

    # Summary text file
    summary_path = args.out_dir / "graph_metrics_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Graph Metrics Summary ({len(rows)} entity types)\n")
        f.write(f"Parameters: {args.n_rules} rules x {args.seeds} seeds x {args.steps} steps\n\n")
        for sz in sorted(set(sizes)):
            subset = [r for r in rows if r["entity_size"] == sz]
            sub_autos = [r["automorphism_count"] for r in subset]
            sub_tri = [r["n_triangles"] for r in subset]
            sub_wedge = [r["n_open_wedges"] for r in subset]
            f.write(
                f"Size {sz}: {len(subset)} types, "
                f"auto=[{min(sub_autos)}, {max(sub_autos)}] mean={sum(sub_autos)/len(sub_autos):.1f}, "
                f"tri=[{min(sub_tri)}, {max(sub_tri)}], "
                f"wedges=[{min(sub_wedge)}, {max(sub_wedge)}]\n"
            )
        try:
            from scipy.stats import spearmanr

            if len(set(sizes)) > 1:
                rho, pval = spearmanr(sizes, autos)
                f.write(f"\nSpearman(size, automorphism_count): rho={rho:.3f}, p={pval:.4g}\n")
        except ImportError:
            pass
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

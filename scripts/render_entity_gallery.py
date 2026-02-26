#!/usr/bin/env python3
"""Render entity gallery: top-k entities by assembly score (a_i × n_i).

Runs block-world simulations, captures canonical graphs for all detected
entities in memory, selects the top-k by a_i × copy_count, and renders
a gallery figure (Figure 2 for the ALIFE paper).

Usage:
    uv run python scripts/render_entity_gallery.py \
        --n-rules 5 --seeds 1 --steps 30 --top-k 4 --out-dir tmp/gallery_smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random

import matplotlib
import networkx as nx
import numpy as np

from alife_discovery.config.constants import ENTITY_SNAPSHOT_INTERVAL
from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.domain.block_world import BlockWorld, generate_block_rule_table
from alife_discovery.domain.entity import (
    canonicalize_entity,
    detect_entities,
    entity_graph_hash,
)
from alife_discovery.metrics.assembly import assembly_index_exact

# Block-type colour map: M=blue, C=green, K=red
_BLOCK_TYPE_COLORS: dict[str, str] = {
    "M": "#3b82f6",
    "C": "#22c55e",
    "K": "#ef4444",
}

# Plotting constants
_NODE_SIZE = 300
_NODE_EDGE_WIDTH = 0.5
_EDGE_COLOR = "#9ca3af"
_EDGE_WIDTH = 1.0
_TITLE_FONTSIZE = 8
_HASH_FONTSIZE = 6
_HASH_COLOR = "#6b7280"
_SUPTITLE_FONTSIZE = 12
_GALLERY_DPI = 150
_SUBPLOT_WIDTH = 3.2
_SUBPLOT_HEIGHT = 3.5
_MAX_COLS = 5


@dataclass
class EntityRecord:
    """Accumulated data for one unique entity type."""

    graph: nx.Graph
    assembly_index: int
    total_copy_count: int
    entity_size: int


def capture_entities(
    *,
    n_rules: int,
    seeds: int,
    steps: int,
    grid_width: int,
    grid_height: int,
    n_blocks: int,
    noise_level: float,
) -> dict[str, EntityRecord]:
    """Run simulations and capture canonical graphs for all entity types.

    Returns a dict mapping entity_hash -> EntityRecord.
    """
    registry: dict[str, EntityRecord] = {}

    for seed in range(seeds):
        for i in range(n_rules):
            cfg = BlockWorldConfig(
                grid_width=grid_width,
                grid_height=grid_height,
                n_blocks=n_blocks,
                noise_level=noise_level,
                steps=steps,
                rule_seed=i,
                sim_seed=seed * n_rules + i,
            )
            rule_table = generate_block_rule_table(cfg.rule_seed)
            rng = Random(cfg.sim_seed)
            world = BlockWorld.create(cfg, rng)

            for step in range(steps):
                world.step(rule_table, noise_level, rng, update_mode=cfg.update_mode)

                if (step + 1) % ENTITY_SNAPSHOT_INTERVAL == 0 or step == steps - 1:
                    entities = detect_entities(world)
                    for entity in entities:
                        g = canonicalize_entity(entity)
                        h = entity_graph_hash(g)
                        if h in registry:
                            registry[h].total_copy_count += 1
                        else:
                            a_i = assembly_index_exact(g)
                            registry[h] = EntityRecord(
                                graph=g,
                                assembly_index=a_i,
                                total_copy_count=1,
                                entity_size=g.number_of_nodes(),
                            )

    return registry


def select_top_k(
    registry: dict[str, EntityRecord],
    k: int,
) -> list[tuple[str, EntityRecord]]:
    """Select top-k entities ranked by a_i × total_copy_count (descending)."""
    scored = [(h, rec, rec.assembly_index * rec.total_copy_count) for h, rec in registry.items()]
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(h, rec) for h, rec, _score in scored[:k]]


def render_entity_subplot(
    ax: matplotlib.axes.Axes,
    graph: nx.Graph,
    *,
    assembly_index: int,
    copy_count: int,
    entity_hash: str,
) -> None:
    """Render a single entity graph into a matplotlib axes."""
    if graph.number_of_nodes() == 0:
        ax.set_visible(False)
        return

    # Layout
    if graph.number_of_nodes() == 1:
        pos = {0: (0.0, 0.0)}
    else:
        pos = nx.kamada_kawai_layout(graph)

    # Node colours from block_type
    node_colors = [
        _BLOCK_TYPE_COLORS.get(graph.nodes[n].get("block_type", "M"), "#888888")
        for n in graph.nodes()
    ]

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=_NODE_SIZE,
        edgecolors="black",
        linewidths=_NODE_EDGE_WIDTH,
    )
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=_EDGE_COLOR, width=_EDGE_WIDTH)

    size = graph.number_of_nodes()
    label = f"$a_i$={assembly_index}  n={copy_count}  size={size}"
    ax.set_title(label, fontsize=_TITLE_FONTSIZE, pad=4)
    ax.set_xlabel(entity_hash[:12] + "...", fontsize=_HASH_FONTSIZE, color=_HASH_COLOR)
    ax.set_axis_off()


def main_gallery(
    *,
    n_rules: int,
    seeds: int,
    steps: int,
    top_k: int,
    grid_width: int,
    grid_height: int,
    n_blocks: int,
    noise_level: float,
    out_dir: Path,
) -> None:
    """Run simulations, select top-k entities, render gallery PDF + CSV."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Capturing entities: {n_rules} rules × {seeds} seeds × {steps} steps ...")
    registry = capture_entities(
        n_rules=n_rules,
        seeds=seeds,
        steps=steps,
        grid_width=grid_width,
        grid_height=grid_height,
        n_blocks=n_blocks,
        noise_level=noise_level,
    )
    print(f"  {len(registry)} unique entity types found")

    top = select_top_k(registry, k=top_k)
    actual_k = len(top)
    if actual_k == 0:
        print("No entities found — skipping gallery.")
        return

    # Layout: grid of subplots
    cols = min(_MAX_COLS, actual_k)
    rows = (actual_k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(_SUBPLOT_WIDTH * cols, _SUBPLOT_HEIGHT * rows))
    if actual_k == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.asarray(axes).flat)

    for idx, (h, rec) in enumerate(top):
        render_entity_subplot(
            axes_flat[idx],
            rec.graph,
            assembly_index=rec.assembly_index,
            copy_count=rec.total_copy_count,
            entity_hash=h,
        )

    # Hide unused subplots
    for idx in range(actual_k, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Top-{actual_k} entities by assembly score ($a_i \\times n_i$)",
        fontsize=_SUPTITLE_FONTSIZE,
        y=1.01,
    )
    fig.tight_layout()

    pdf_path = out_dir / "entity_gallery.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=_GALLERY_DPI)
    plt.close(fig)
    print(f"Gallery figure: {pdf_path}")

    # Write metadata CSV
    csv_path = out_dir / "entity_gallery_meta.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "entity_hash",
                "assembly_index",
                "copy_count",
                "entity_size",
                "score",
            ],
        )
        writer.writeheader()
        for rank, (h, rec) in enumerate(top, 1):
            writer.writerow(
                {
                    "rank": rank,
                    "entity_hash": h,
                    "assembly_index": rec.assembly_index,
                    "copy_count": rec.total_copy_count,
                    "entity_size": rec.entity_size,
                    "score": rec.assembly_index * rec.total_copy_count,
                }
            )
    print(f"Metadata CSV: {csv_path}")


def _positive_int(value: str) -> int:
    """Argparse type for strictly positive integers."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render entity gallery (Figure 2)")
    p.add_argument("--n-rules", type=_positive_int, default=100)
    p.add_argument("--seeds", type=_positive_int, default=3)
    p.add_argument("--steps", type=_positive_int, default=200)
    p.add_argument("--top-k", type=_positive_int, default=10)
    p.add_argument("--grid-width", type=_positive_int, default=20)
    p.add_argument("--grid-height", type=_positive_int, default=20)
    p.add_argument("--n-blocks", type=_positive_int, default=30)
    p.add_argument("--noise-level", type=float, default=0.01)
    p.add_argument("--out-dir", type=Path, default=Path("data/entity_gallery"))
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Fail fast on invalid argument combinations."""
    if args.n_blocks > args.grid_width * args.grid_height:
        sys.exit(
            f"error: --n-blocks ({args.n_blocks}) exceeds grid capacity "
            f"({args.grid_width}×{args.grid_height}={args.grid_width * args.grid_height})"
        )
    if not 0.0 <= args.noise_level <= 1.0:
        sys.exit(f"error: --noise-level must be in [0.0, 1.0], got {args.noise_level}")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    main_gallery(
        n_rules=args.n_rules,
        seeds=args.seeds,
        steps=args.steps,
        top_k=args.top_k,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        n_blocks=args.n_blocks,
        noise_level=args.noise_level,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

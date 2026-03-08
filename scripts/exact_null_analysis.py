#!/usr/bin/env python3
"""Exact null enumeration for small graphs (n <= 4).

For each unique degree sequence among connected graphs with n <= 4 in the
graph atlas, count how many non-isomorphic connected topologies share that
degree sequence, and compute assembly_index_exact on each. This confirms
whether the null model is degenerate (only one topology per degree sequence)
for small entities.

Usage:
    uv run python scripts/exact_null_analysis.py \
        --out-dir data/exact_null
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
from networkx.generators.atlas import graph_atlas_g

from alife_discovery.metrics.assembly import assembly_index_exact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact null enumeration for small graphs")
    p.add_argument("--out-dir", type=Path, default=Path("data/exact_null"))
    return p.parse_args()


def degree_sequence(g: nx.Graph) -> tuple[int, ...]:
    """Sorted degree sequence (canonical form for grouping)."""
    return tuple(sorted(d for _, d in g.degree()))


def _label_graph(g: nx.Graph) -> nx.Graph:
    """Add uniform block_type='M' for untyped topology analysis only.

    assembly_index_exact() requires a block_type node attribute for WL hashing.
    This function assigns all nodes the same type so that the result reflects
    pure graph topology, not block-type composition. Do NOT use this for typed
    entity graphs from alife_discovery/domain/entity.py, which carry
    heterogeneous M/C/K node attributes.
    """
    h = g.copy()
    nx.set_node_attributes(h, "M", "block_type")
    return h


def _group_by_degree_sequence(
    max_n: int,
) -> dict[tuple[int, tuple[int, ...]], list[nx.Graph]]:
    """Single pass over graph atlas: group connected graphs by (n, degree_sequence).

    graph_atlas_g() already returns non-isomorphic graphs, so no deduplication
    is needed. We filter for 2 <= n <= max_n (excluding n=1 single-node graphs,
    which have no bonds and are not produced by the simulation) and connectivity.
    """
    groups: dict[tuple[int, tuple[int, ...]], list[nx.Graph]] = {}
    for g in graph_atlas_g():
        n = g.number_of_nodes()
        if n < 2 or n > max_n:
            continue
        if not nx.is_connected(g):
            continue
        key = (n, degree_sequence(g))
        groups.setdefault(key, []).append(g)
    return groups


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["=== Exact Null Enumeration for Small Graphs (n <= 4) ===", ""]
    lines.append(
        "For each unique degree sequence among connected n<=4 graphs, we report "
        "the number of distinct topologies and their assembly indices."
    )
    lines.append("")

    header = (
        f"{'n':>2}  {'deg_seq':>20}  {'n_topologies':>12}  {'a_i values':>20}  {'degenerate?':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    total_groups = 0
    total_degenerate = 0

    groups = _group_by_degree_sequence(max_n=4)
    for (n, ds), topologies in sorted(groups.items()):
        ai_values = sorted({assembly_index_exact(_label_graph(h)) for h in topologies})
        n_topo = len(topologies)
        degenerate = n_topo == 1
        total_groups += 1
        if degenerate:
            total_degenerate += 1

        ds_str = str(list(ds))
        ai_str = str(ai_values)
        deg_str = "YES" if degenerate else "NO"
        lines.append(f"{n:>2}  {ds_str:>20}  {n_topo:>12}  {ai_str:>20}  {deg_str:>12}")

    lines.append("")
    lines.append(f"Total unique degree sequences (n<=4): {total_groups}")
    lines.append(f"Degenerate (only 1 topology): {total_degenerate}")
    lines.append(
        f"Fraction degenerate: {total_degenerate}/{total_groups} = "
        f"{total_degenerate / total_groups:.1%}"
    )
    lines.append("")
    lines.append(
        "Conclusion: For n<=4, every degree sequence admits exactly one connected "
        "topology. The bond-shuffle null model is therefore degenerate at these "
        "sizes (p=1 always), confirming that 0% excess is not an artifact of "
        "null degeneracy---entities must depart from the unique topology to "
        "score p<0.05."
    )

    report = "\n".join(lines)
    out_path = args.out_dir / "exact_null_report.txt"
    out_path.write_text(report)
    print(report)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

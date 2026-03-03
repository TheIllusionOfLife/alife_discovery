"""Graph complexity metrics: automorphism count and typed motif census.

Complements Assembly Theory metrics with structural measures that capture
symmetry and local motif composition of entity graphs.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher


def graph_automorphism_count(g: nx.Graph) -> int:
    """Count the number of label-preserving automorphisms of graph ``g``.

    Uses NetworkX ``GraphMatcher`` for self-isomorphism enumeration.
    Safe for entity graphs with n <= 16 nodes.

    Returns 1 for the empty graph (trivial automorphism group).
    """
    if g.number_of_nodes() == 0:
        return 1

    def node_match(n1: dict[str, Any], n2: dict[str, Any]) -> bool:
        return n1.get("block_type") == n2.get("block_type")

    gm = GraphMatcher(g, g, node_match=node_match)
    count = 0
    for _ in gm.isomorphisms_iter():
        count += 1
    return max(count, 1)


def typed_motif_census(g: nx.Graph) -> dict[str, Any]:
    """Compute 3-node subgraph motif census with type labels.

    Returns:
        Dictionary with:
        - ``triangles``: number of triangles (3-cliques)
        - ``open_wedges``: number of open wedges (paths of length 2)
        - ``typed_triangles``: list of sorted type-label tuples for each triangle
        - ``typed_wedges``: list of sorted type-label tuples for each open wedge
    """
    triangles: list[tuple[str, ...]] = []
    open_wedges: list[tuple[str, ...]] = []

    nodes = list(g.nodes())
    if len(nodes) < 3:
        return {
            "triangles": 0,
            "open_wedges": 0,
            "typed_triangles": [],
            "typed_wedges": [],
        }

    for u, v, w in combinations(nodes, 3):
        edges = sum(1 for a, b in [(u, v), (v, w), (u, w)] if g.has_edge(a, b))
        if edges == 3:
            # Triangle
            types = tuple(sorted(g.nodes[n].get("block_type", "?") for n in (u, v, w)))
            triangles.append(types)
        elif edges == 2:
            # Open wedge
            types = tuple(sorted(g.nodes[n].get("block_type", "?") for n in (u, v, w)))
            open_wedges.append(types)

    return {
        "triangles": len(triangles),
        "open_wedges": len(open_wedges),
        "typed_triangles": triangles,
        "typed_wedges": open_wedges,
    }
